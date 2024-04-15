import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.models import SimVP_Model
from openstl.utils import reduce_tensor
from .simvp import SimVP

import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weights,thres):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights
        self.thres = thres
    def forward(self, y_pred, y_true):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        # self.thres = [-10.716510772705078,-6.218186950683593,-1.7198631286621087,2.778460693359376,7.276784515380861,11.775108337402344]
        # self.weights  = 1/[100279308,11304253,2046067,6644,1391]
        mask0 = y_true<self.thres[1]
        mask1 = (y_true>self.thres[1])*(y_true<self.thres[2])
        mask2 = (y_true>self.thres[2])*(y_true<self.thres[3])
        mask3 = (y_true>self.thres[3])*(y_true<self.thres[4])
        mask4 = y_true>self.thres[4]
        squared_errors = torch.square(y_pred - y_true)
        abs_errors = torch.abs(y_pred - y_true)
        squared_errors = squared_errors+abs_errors
        loss = mask0*squared_errors*self.weights[0]+mask1*squared_errors*self.weights[1]+mask2*squared_errors*self.weights[2]+mask3*squared_errors*self.weights[3]+mask4*squared_errors*self.weights[4]
        # weighted_squared_errors = squared_errors * self.weights.unsqueeze(1)
        loss = torch.mean(loss)
        return loss

class TAU(SimVP):
    r"""TAU

    Implementation of `Temporal Attention Unit: Towards Efficient Spatiotemporal 
    Predictive Learning <https://arxiv.org/abs/2206.12126>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        SimVP.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()
        # thres = torch.tensor([-8.562353134155273,-6.098530673980713,-3.6347082138061526,-1.170885753631592,1.2929367065429687,3.7567591667175293]).to(self.device)
        # weights  = 10000000.0/torch.tensor([56017088,32376370,19246055,5844945,153205]).to(self.device)
        # weights  = torch.tensor([1,1,1,2,8]).to(self.device)

        weights=10000000.0/torch.tensor([[56017088,32376370,19246055,5844945,153205],
        [100279308,11304253,2046067,6644,1391],
        [104234835,6252654,3140061,7514,2599],
        [64469971,48003036,1091871,71839,946],
        [64911287,10627441,2755812,281476,10271]]).to(self.device)

        # thres
        thres = torch.tensor([[-8.562353134155273,-6.098530673980713,-3.6347082138061526,-1.170885753631592,1.2929367065429687,3.7567591667175293],
        [-10.716510772705078,-6.218186950683593,-1.7198631286621087,2.778460693359376,7.276784515380861,11.775108337402344],
        [-10.40748119354248,-6.205681991577149,-2.0038827896118168,2.197916412353515,6.399715614318847,10.60151481628418],
        [-10.218382835388184,-6.865275859832764,-3.5121688842773438,-0.15906190872192383,3.194045066833496,6.547152042388916],
        [-0.6048222780227661,1.3249369621276856,3.2546962022781374,5.18445544242859,7.114214682579041,9.043973922729492]]).to(self.device)

        self.criterion1 = WeightedMSELoss(weights[0],thres[0])
        self.criterion2 = WeightedMSELoss(weights[1],thres[1])
        self.criterion3 = WeightedMSELoss(weights[2],thres[2])
        self.criterion4 = WeightedMSELoss(weights[3],thres[3])
        self.criterion5 = WeightedMSELoss(weights[4],thres[4])
    def _build_model(self, args):
        return SimVP_Model(**args).to(self.device)
    
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
            batch_x=batch_x.float()
            batch_y=batch_y.float()
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            # with self.amp_autocast():
            pred_y = self._predict(batch_x)

            # loss = self.criterion(pred_y, batch_y)+self.criterion1(pred_y[:,:,0], batch_y[:,:,0])
            loss = self.criterion(pred_y, batch_y) + self.args.alpha * self.diff_div_reg(pred_y, batch_y)*50000
            # if epoch>=0:
            #     loss1 = self.criterion(pred_y[:,0], batch_y[:,0]) *0.1+self.criterion(pred_y[:,1], batch_y[:,1]) *0.1+self.criterion(pred_y[:,2], batch_y[:,2]) *0.1+self.criterion(pred_y[:,3], batch_y[:,3]) *0.1+\
            #             self.criterion(pred_y[:,4], batch_y[:,4]) *0.15+self.criterion(pred_y[:,5], batch_y[:,5]) *0.15+self.criterion(pred_y[:,6], batch_y[:,6]) *0.15+self.criterion(pred_y[:,7], batch_y[:,7]) *0.15+\
            #             self.criterion(pred_y[:,8], batch_y[:,8]) *0.20+self.criterion(pred_y[:,9], batch_y[:,9]) *0.2+self.criterion(pred_y[:,10], batch_y[:,10]) *0.2+self.criterion(pred_y[:,11], batch_y[:,11]) *0.2+\
            #             self.criterion(pred_y[:,12], batch_y[:,12]) *0.25+self.criterion(pred_y[:,13], batch_y[:,13]) *0.25+self.criterion(pred_y[:,14], batch_y[:,14]) *0.25+self.criterion(pred_y[:,15], batch_y[:,15]) *0.25+\
            #             self.criterion(pred_y[:,16], batch_y[:,16]) *0.30+self.criterion(pred_y[:,17], batch_y[:,17]) *0.30+self.criterion(pred_y[:,18], batch_y[:,18]) *0.30+self.criterion(pred_y[:,19], batch_y[:,19]) *0.30
            #     loss = loss1/4
            # + self.args.alpha * self.diff_div_reg(pred_y, batch_y)*50000
            # if epoch>0:
            #     loss1 = self.criterion(pred_y[:,:,1], batch_y[:,:,1]) 
            #     loss = loss1*5
                # + self.args.alpha * self.diff_div_reg(pred_y[:,:,0], batch_y[:,:,0])*50000
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
            # print(loss.item())
            # print(loss1.item())
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
