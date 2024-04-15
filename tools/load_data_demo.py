
import torch
import glob
import os

file1 = '/home/convlstm_predict/2023tianchi_weather_prediction/weather_round1_test/output/output/020.pt'
file2 = '/home/convlstm_predict/2023tianchi_weather_prediction/weather_round1_test/output/t2m/020.pt'
tmp_data1 = torch.load(file1)
tmp_data2 = torch.load(file2)

print("end")