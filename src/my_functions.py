import numpy as np
import torch
import sys
import h5py
import pickle
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

def get_lb_data(LB_DATA_PATH):
    with open(LB_DATA_PATH, "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        simdata = pickle.load(file)
        rho_act_all = np.array(simdata["rho_act_all"])
        rho_in_all = np.array(simdata["rho_in_all"])
        # dt_coarse = simdata["dt_data"]
        del simdata
    # print(rho_act_all.shape)
    # print(rho_in_all.shape)
    # print(type(rho_in_all))
    # print(rho_in_all.dtype)
    rho_act_all_1 = np.expand_dims(rho_act_all, axis=2)
    rho_in_all_1 = np.expand_dims(rho_in_all, axis=2)
    # print(np.shape(rho_act_all_1))
    # print(np.shape(rho_in_all_1))
    rho_all = np.concatenate((rho_act_all_1, rho_in_all_1), axis=2)
    # print(np.shape(rho_all))
    return rho_all

def reshape_input_data(data):
    print("Shape of data:", data.shape)

    data_1 = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])
    print("Shape of data_1:", data_1.shape)

    data_final = data_1.reshape(data_1.shape[0], data_1.shape[1]*data_1.shape[2])
    print("Shape of data_final:", data_final.shape)
    return data_final

def create_inout_sequences(input_data, input_sequence_length):
    x_train = [input_data[i:i+input_sequence_length] for i in range(input_data.shape[0] - input_sequence_length - 1)]
    y_train = [input_data[i+input_sequence_length] for i in range(input_data.shape[0] - input_sequence_length - 1)]
    # print(len(x_train))
    # print(len(y_train))
    # print(x_train[0].shape)
    # print(y_train[0].shape)
    x_train = torch.stack(x_train)
    # x_train = torch.reshape(x_train,(input_sequence_length,-1,2))
    print("Shape of x_train",x_train.shape)
    y_train = torch.stack(y_train)
    print("Shape of y_train:",y_train.shape)
    return x_train, y_train

def create_subsampled_sequences(data,sequence_length,sampling_freq):
    x_train = [data[i:i+sampling_freq*sequence_length][::sampling_freq] for i in range(data.shape[0] - sequence_length*sampling_freq - 1)]
    y_train = [data[i+sampling_freq*sequence_length] for i in range(data.shape[0] - sequence_length*sampling_freq - 1)]
    # print(len(x_train))
    x_train = torch.stack(x_train)
    print("Shape of x_train",x_train.shape)
    y_train = torch.stack(y_train)
    print("Shape of y_train:",y_train.shape)
    return x_train, y_train

class AE(nn.Module):
    def __init__(self,mode='encoder'):
        super().__init__()
        self.mode=mode
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(202, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 2)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 202),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        if self.mode =='encoder':
            encoded = self.encoder(x)
            decoded  = self.decoder(encoded)
            return decoded, encoded
        if self.mode == 'decoder_only':
            decoded  = self.decoder(x)
            return decoded


class LSTMnet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMnet,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, input_size)


    def forward(self, x,hidden=None):
        if hidden==None:
            self.hidden = (torch.zeros(self.num_layers,x.size(0) ,self.hidden_size),torch.zeros(self.num_layers,x.size(0),self.hidden_size))
        else:
            self.hidden = hidden
        lstm_out, self.hidden = self.lstm(x,self.hidden)
        # print("shape of lstm_out:", lstm_out.shape)
        # print("shape of hidden:",self.hidden[0].shape)
        predictions = self.out(lstm_out)
        return predictions[:,-1]

def iterative_forecasting(input_sequence,prediction_horizon,sequence_length,model):
    input=input_sequence
    outputs=input
    for i in range(prediction_horizon):
        input_=input.view(1,input.shape[0],input.shape[1])
        # print(input_.shape)
        output = model(input_)
        # print(output.shape)
        input=torch.cat((input[-sequence_length+1:],output),0)
        outputs=torch.cat((outputs,output),0)
    return outputs