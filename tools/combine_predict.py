
import torch
import glob
import os
varibles = ['t2m','u10','v10','msl','tp']
data_dir = '/home/convlstm_predict/2023tianchi_weather_prediction/weather_round1_test/output'
output_data = torch.zeros((300,20,5,161,161))

for var_id in range(len(varibles)):
    var = varibles[var_id]
    var_data = []
    var_data_files = glob.glob(os.path.join(data_dir,var,"*.pt"))
    var_data_files = sorted(var_data_files)
    for file_id in range(len(var_data_files)):
        file = var_data_files[file_id]
        tmp_data = torch.load(file)
        output_data[file_id,:,var_id,:,:] = tmp_data[:,0,:,:]
# output_data = output_data.to(torch.float16)

save_dir = '/home/convlstm_predict/2023tianchi_weather_prediction/weather_round1_test/output/output'
for i in range(len(output_data)):
    pred_pt = output_data[i]
    pred_pt = pred_pt.to(torch.float16)
    torch.save(pred_pt,os.path.join(save_dir,str(i).zfill(3)+".pt"))
print("end")