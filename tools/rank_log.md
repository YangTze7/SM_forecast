model: TAU
input: 5 var
epoch: 100epoch

score:0.1777
t2m:0.0000
u10:0.2328
v10:0.1512
msl:0.3469
tp:0.1576



model: TAU
input: 10 var
epoch: 100epoch
aug:no
score:0.2836
t2m:0.2041
u10:0.3201
v10:0.2355
msl:0.4606
tp:0.1978
loss: t2m loss + total loss



model:tau
input: 10 var
single var infer
weighted loss
score:0.1748
t2m:0.2047
u10:0.1844
v10:0.0632
msl:0.3790
tp:0.0427

model:simvp
input: 10 var
single var infer
score:0.2148
t2m:0.0010
u10:0.2971
v10:0.1954
msl:0.4036
tp:0.1766


model:tau
input: 10 var
single car input
single var infer

score:0.1957
t2m:0.0000
u10:0.2822
v10:0.1476
msl:0.3835
tp:0.1652


model:predRNN
input: 10 var
multi var input
single var infer

score:0.1957
