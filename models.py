import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv

class EdgeNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=1, n_iters=1,aggr='add'):
        super(EdgeNet, self).__init__()
        convnn = nn.Sequential(nn.Linear(2*(hidden_dim + input_dim), hidden_dim),
                               nn.Sigmoid()
        )
        self.n_iters = n_iters
        
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.edgenetwork = nn.Sequential(nn.Linear(2*(hidden_dim+input_dim),
                                                   2*hidden_dim),
                                         nn.Sigmoid())

        self.outputnet = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                       nn.Sigmoid())

        self.nodenetwork = EdgeConv(nn=convnn,aggr=aggr)

    def forward(self, data):
        X = data.x
        H = self.inputnet(X)
        data.x = torch.cat([H,X],dim=-1)
        for i in range(self.n_iters):
            H = self.nodenetwork(data.x,data.edge_index)
            data.x = torch.cat([H,X],dim=-1)
        sum_H = torch.mean(H, dim=0)
        #row,col = data.edge_index        
        #output = self.edgenetwork(torch.cat([data.x[row],data.x[col]],dim=-1)).squeeze(-1)
        output = self.outputnet(sum_H)
        return output

import keras
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv1D, SpatialDropout1D, GRU, Concatenate, Flatten

def dense(nfeatures, ncsc, ncscfeatures, nlabels):

    inputs = Input(shape=(nfeatures,), name = 'input')  
    inputs_csc = Input(shape=(ncsc,ncscfeatures,), name='input_csc')
    flat = Flatten()(inputs_csc)
    x = Concatenate()([inputs, flat])
    x = BatchNormalization(name='bn_1')(x)
    x = Dense(64, name = 'dense_1', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(32, name = 'dense_2', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(32, name = 'dense_3', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    
    outputs = Dense(nlabels, name = 'output', activation='sigmoid')(x)
    keras_model = Model(inputs=[inputs,inputs_csc], outputs=outputs)

    return keras_model


def dense_conv1d_gru(nfeatures, ncsc, ncscfeatures, nlabels):

    inputs = Input(shape=(nfeatures,), name = 'input')  
    x = BatchNormalization(name='bn_1')(inputs)
    x = Dense(64, name = 'dense_1', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(32, name = 'dense_2', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(32, name = 'dense_3', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    
    inputs_csc = Input(shape=(ncsc,ncscfeatures,), name='input_csc')
    y = BatchNormalization(name='bn_2')(inputs_csc)
    y = Conv1D(filters=64, kernel_size=(5,), strides=(1,), padding='same', use_bias=False, name='conv1d_1', activation = 'relu')(y)
    y = SpatialDropout1D(rate=0.1)(y)
    y = Conv1D(filters=64, kernel_size=(5,), strides=(1,), padding='same', use_bias=False, name='conv1d_2', activation = 'relu')(y)
    y = SpatialDropout1D(rate=0.1)(y)
    y = GRU(150,go_backwards=True,implementation=2,name='gru_1')(y)
    y = Dropout(rate=0.1)(y)

    concat = Concatenate()([x,y])

    outputs = Dense(nlabels, name = 'output', activation='sigmoid')(concat)
    keras_model = Model(inputs=[inputs,inputs_csc], outputs=outputs)

    return keras_model
