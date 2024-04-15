import time
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import AverageMeter
import torch.nn.functional as F
from openstl.models import SimVP_Model
from openstl.utils import reduce_tensor
from .base_method import Base_method


class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        return SimVP_Model(**args).to(self.device)

    def _predict(self, batch_x, batch_y=None, **kwargs):
        """Forward the model"""
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            # dem = batch_x[:,:,5:6,:,:]
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                # cur_seq = torch.concat([cur_seq,dem],dim=2)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                # cur_seq = torch.concat([cur_seq,dem],dim=2)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        # pred_y = self.model(batch_x)

        # pred_y = pred_y[:,:,0:5,:,:]
        return pred_y
    
    def diff_div_reg(self, pred_y, batch_y, tau=0.1, eps=1e-12):
        B, T, C = pred_y.shape[:3]
        if T <= 2:  return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        return loss_gap.mean()
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            with self.amp_autocast():
                pred_y = self._predict(batch_x)
                # loss = self.criterion(pred_y, batch_y)
            self.args.alpha = 0.1
            loss = self.criterion(pred_y, batch_y) + self.args.alpha * self.diff_div_reg(pred_y, batch_y)*50000

            # if epoch>=0:
            #     loss1 = self.criterion(pred_y[:,:,0], batch_y[:,:,0]) + self.args.alpha * self.diff_div_reg(pred_y[:,:,0], batch_y[:,:,0])*50000
            #     loss = loss1*5
            # if epoch>30:
            #     loss2 = self.criterion(pred_y[:,:,1], batch_y[:,:,1]) + self.args.alpha * self.diff_div_reg(pred_y[:,:,1], batch_y[:,:,1])*50000
            #     loss = loss2*5
            # if epoch>40:
            #     loss3 = self.criterion(pred_y[:,:,2], batch_y[:,:,2]) + self.args.alpha * self.diff_div_reg(pred_y[:,:,2], batch_y[:,:,2])*50000
            #     loss = loss3*5
            # if epoch>50:
            #     loss4 = self.criterion(pred_y[:,:,3], batch_y[:,:,3]) + self.args.alpha * self.diff_div_reg(pred_y[:,:,3], batch_y[:,:,3])*50000
            #     loss = loss4*5
            # if epoch>60:
            #     loss5 = self.criterion(pred_y[:,:,4], batch_y[:,:,4]) + self.args.alpha * self.diff_div_reg(pred_y[:,:,4], batch_y[:,:,4])*50000
            #     loss = loss5*5
                # loss = loss+loss1*5
            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta