import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class FlowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(FlowLSTM, self).__init__()
        # build your model here
        # your input should be of dim (batch_size, seq_len, input_size)
        # your output should be of dim (batch_size, seq_len, input_size) as well
        # since you are predicting velocity of next step given previous one
        
        # feel free to add functions in the class if needed
        
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.dropout=dropout
        
        self.lstm1 = nn.LSTMCell(self.input_size,self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size )
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    # forward pass through LSTM layer
    def forward(self, x):
        '''
        input: x of dim (batch_size, 19, 17)
        '''
        # define your feedforward pass

        outputs = []
        self.x=x
        h_t= torch.zeros(x.size(0),self.hidden_size, dtype=torch.float)
        c_t= torch.zeros(x.size(0), self.hidden_size, dtype=torch.float)
        h_t2= torch.zeros(x.size(0), self.hidden_size, dtype=torch.float)
        c_t2 = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float)
        x1=x.permute(1, 0, 2)
        for i in range(x.size(1)):
            h_t, c_t = self.lstm1(x1[i], (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # h_t2 = self.dropout(h_t2)
            nn.Dropout(self.dropout)
            output = self.linear(h_t2)
            outputs += [output]
    #     # for i in range(future):
    #     #     h_t, c_t = self.lstm1(output, (h_t, c_t))
    #     #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
    #     #     output = self.linear(h_t2)
    #     #     outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        # print('outputs',outputs)
        return outputs


    # # forward pass through LSTM layer for testing
    def test(self, x):
        '''
        input: x of dim (batch_size, 17)
        '''
        outputs = []
        self.x=x
        h_t = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float)
        c_t = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float)
        h_t2 = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float)
        c_t2 = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float)
        # define your feedforward pass
        output=x
        for i in range(19):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
