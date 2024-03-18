import numpy as np
import torch
import sys
import h5py
import pickle
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import os
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')
# Path for various functions

# module_path= os.path.abspath(os.path.join('../HiTS/AEHITSCODE/src/'))
module_path= os.path.abspath(os.path.join('../../AEHITSCODE/src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
import AE

#%%
module_path2= os.path.abspath(os.path.join('../HiTS/AEHITSCODE/LEDdata/'))
if module_path2 not in sys.path:
    sys.path.append(module_path2)
system = 'FHN'
from my_functions import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = os.path.join('../RSmodels/', system)

#%% Gen the simulation data from LBM
LB_DATA_PATH = "lattice_boltzmann_fhn_test.pickle"
data= get_lb_data(LB_DATA_PATH)
data_final = reshape_input_data(data)

scaler = MinMaxScaler()
scaler.fit(data_final)
data_scaled = scaler.transform(data_final)

a = data_scaled.shape[0]/2
b = data_scaled.shape[0]*(5/6)

training_data = data_scaled[0:int(a)]
validation_data = data_scaled[int(a):int(b)]
test_data = data_scaled[int(b)::]
print(training_data.shape)
print(validation_data.shape)
print(test_data.shape)

ts=10019
n=202
t_train=5000
training_data=training_data.reshape(3,ts,n)
validation_data=validation_data.reshape(2,ts,n)
training_data=training_data[:,:t_train,:]
validation_data=validation_data[:,:t_train,:]
training_data=training_data.reshape(3*t_train,n)
validation_data=validation_data.reshape(2*t_train,n)
training_data_T = torch.tensor(training_data).float()
train_loader = torch.utils.data.DataLoader(dataset = training_data_T, batch_size=32, shuffle=True)

for one in train_loader:
    print(one.shape)
    # print(one)
    break


#%% Load trained AE model
model = AE()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
model.load_state_dict(torch.load('../RSmodels/FHN/aeweights3.pt'))

model.eval()
with torch.no_grad():
    output_train_T, latent_train_T = model(training_data_T)
output_train=output_train_T.detach().numpy()
output_unscalled=scaler.inverse_transform(output_train)

# evaluate the trained AE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
MSE = mean_squared_error(training_data,output_train)
r2 = r2_score(training_data,output_train)

from sklearn.metrics import r2_score
validation_data_T = torch.tensor(validation_data).float()
test_data_T = torch.tensor(test_data).float()

# Validation Data
model.eval()
with torch.no_grad():
    output_val_T, latent_val_T = model(validation_data_T)

output_val=output_val_T.detach().numpy()
output_unscalled_val=scaler.inverse_transform(output_val)
MSE_val = mean_squared_error(validation_data,output_val)

# Testing Data
model.eval()
with torch.no_grad():
    output_test_T, latent_test_T = model(test_data_T)

output_val=output_val_T.detach().numpy()
output_test=output_test_T.detach().numpy()
output_unscalled_test=scaler.inverse_transform(output_test)


MSE_test = mean_squared_error(test_data,output_test)
print(MSE_test)

r2_val = r2_score(validation_data,output_val)
r2_test = r2_score(test_data,output_test)
print(r2_test)

fig = plt.figure()
ax = fig.add_axes([0,0,0.5,0.5])
this = ['Training', 'Validation', 'Testing']
that = [MSE,MSE_val, MSE_test]
ax.bar(this,that)
ax.set_ylabel('Mean Square Error')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,0.5,0.5])
this = ['Training', 'Validation','Testing']
that = [r2,r2_val, r2_test]
ax.bar(this,that)
ax.set_ylabel('R2')
plt.ylim([0.99, 1])
plt.show()

#%% Resnet training
latent_train_unscalled = latent_train_T
latent_test_unscalled = latent_test_T
latent_val_unscalled = latent_val_T

tspan=5000
z=2
ntrain=3
nval=2
ntest=1
np.random.seed(2)  # for reproduction
dt = 0.01  #0.001
train_steps = tspan  # at least equal to the largest step size
val_steps =tspan
test_steps = tspan  # t=20
t = np.linspace(0, (train_steps-1)*dt, train_steps)

#Data for latent dimension
# z_train=latent_train_unscalled.reshape(ntrain,tspan,z)
# z_valid=latent_val_unscalled.reshape(nval,tspan,z)
# z_test=latent_test_unscalled[:10000].reshape(ntest,2*tspan,z)

z_train=latent_train_unscalled
z_valid=latent_val_unscalled
z_test=latent_test_unscalled[:10000]

import ResNet as net
from utils import *


# training
model_prefix = 'FHN'
n_forward=5
max_epoch=25000
L1=128
L2=256
L3=512
L4=1024
L5=2048
arch=[z,L1,L1,L1,z]
step_sizes = list()
for i in range(11):
    step_size = 2**i  #exponential function
    print(step_size)
    print(step_size * dt)
    step_sizes.append(step_size)
models = []
n_steps = z_train.shape[0] - 1  # number of forward steps
for step_size in step_sizes:
    m = int(np.ceil(n_steps/(step_size*n_forward)))
    pdata = np.zeros((m, step_size*n_forward+1, z_train.shape[1]))
    for i in range(m):
        start_idx = i*step_size*n_forward
        end_idx = start_idx + step_size*n_forward + 1
        tmp = z_train[start_idx:end_idx, :]
        pdata[i, :tmp.shape[0], :] = tmp
    pdata_valid = np.zeros((m, step_size * n_forward + 1, z_valid.shape[1]))
    for i in range(m):
        start_idx = i * step_size * n_forward
        end_idx = start_idx + step_size * n_forward + 1
        tmp = z_valid[start_idx:end_idx, :]
        pdata_valid[i, :tmp.shape[0], :] = tmp
    datasets = net.DataSet(pdata, pdata_valid, z_test[np.newaxis, :], dt, step_size, n_forward)
    print('MODEL: '+model_prefix+'_D{}'.format(step_size))
    model = net.ResNet(arch=arch, dt=dt, step_size=step_size)
    model.train_net(datasets, max_epoch=max_epoch, batch_size=50, lr=1e-3,
                    model_path=os.path.join(model_dir, model_prefix+'_D{}.pt'.format(step_size)))
    models.append(model)



# load the data to dataset object
#datasets = list()

# print('Dt\'s: ')
#a=[10,20,30,60,100,200,400,600,800]

    # datasets.append(DataSet(z_train, z_valid, z_test, dt, step_size=step_size, n_forward=19))

# models = list()
# max_epoch=15000
# L1=128
# L2=256
# L3=512
# L4=1024
# L5=2048
# start_time1 = time.time()
# for (step_size, dataset) in zip(step_sizes, datasets):
#     # set up the network
#     model_name = 'model3_D{}.pt'.format(step_size)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     modelresnet = net.ResNet(arch=[z,L1,L1,L1,z], dt=dt, step_size=step_size)
#     # training
#     print('training model3_D{} ...'.format(step_size))
#     modelresnet.train_net(dataset, max_epoch=max_epoch, batch_size=ntrain, lr=1e-3, model_path=os.path.join(model_dir, model_name))
#     models.append(modelresnet)
# end_time1 = time.time()
# offline_time=end_time1-start_time1
# print('offline time elaspsed is',offline_time)
print('models trained successfully!')



# #%% Forecasting
# n_steps = test_data.shape[0] - 1
# n_steps=n_steps-19
# criterion = torch.nn.MSELoss(reduction='none')
# preds_mse = list()
# times = list()
# z_hits_uni=[]
# print('uniscale forecasting...')
# for model in models:
#     start = time.time()
#     z_hits = model.uni_scale_forecast(torch.tensor(z_test[0, :]).float(), n_steps=n_steps)
#     z_hits_uni.append(z_hits)
#     end = time.time()
#     times.append(end - start)
#     preds_mse.append(criterion(torch.tensor(z_test[1:, :]).float(), z_hits).mean(-1))
# print('prediction recorded!')
#
# # model selections
# start_idx = 0
# end_idx = len(models)
# best_mse = 1e+5  # 1e+5
#
# # choose the largest time step
# for i in tqdm(range(len(models))):
#     z_hits = net.vectorized_multi_scale_forecast(torch.tensor(z_valid[:, 0, :]).float(), n_steps=test_steps - 1,
#                                                  models=models[:len(models) - i]).to(device)
#     mse = criterion(torch.tensor(z_valid[:, 1:, :]).float(), z_hits).mean().item()
#     if mse <= best_mse:
#         end_idx = len(models) - i
#         best_mse = mse
#
# # choose the smallest time step
# for i in tqdm(range(end_idx)):
#     z_hits = net.vectorized_multi_scale_forecast(torch.tensor(z_valid[:, 0, :]).float(), n_steps=test_steps - 1,
#                                                  models=models[i:end_idx]).to(device)
#     mse = criterion(torch.tensor(z_valid[:, 1:, :]).float(), z_hits).mean().item()
#     if mse <= best_mse:
#         start_idx = i
#         best_mse = mse
#
# print('use models {} - {}.'.format(start_idx, end_idx))
# models1 = models[start_idx:end_idx]
# # models11=models[1:7]
# # multiscale time-stepping with NN
# start_time = time.time()
# z_hits = net.vectorized_multi_scale_forecast(torch.tensor(z_test[:, 0, :]).float(), n_steps=2 * test_steps - 1,
#                                              models=models1).to(device)
# end_time = time.time()
# online_time = end_time - start_time
# print('online time elaspsed is', online_time)
# multiscale_preds_mse = criterion(torch.tensor(z_test[:, 1:, :]).float(), z_hits).mean(-1)
# multiscale_err = multiscale_preds_mse.mean(0).cpu().detach().numpy()
#
# # visualize forecasting error at each time step
# t = np.linspace(0, (2 * train_steps - 1) * dt, 2 * train_steps)
# fig = plt.figure(figsize=(20, 6))
# colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(datasets))))
# for k in range(len(preds_mse)):
#     err = preds_mse[k]
#     mean = err.mean(0).cpu().detach().numpy()
#     rgb = next(colors)
#     plt.plot(t[:-1], np.log10(mean), linestyle='-', color=rgb, linewidth=3.0, alpha=0.5,
#              label='$\Delta\ t$={}'.format(step_sizes[k] * dt))
# plt.plot(t[:-1], np.log10(multiscale_err), linestyle='-', color='k', linewidth=4.0, label='multiscale')
# plt.legend(fontsize=20, loc='upper right')
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
#
# plt.show()