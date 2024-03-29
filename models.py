import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, global_mean_pool

class EdgeNet(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=32, output_dim=1, n_iters=1,aggr='add'):
        super(EdgeNet, self).__init__()
        convnn = nn.Sequential(nn.Linear(2*(hidden_dim + input_dim), 2*hidden_dim),
                               nn.ReLU(),
                               nn.Linear(2*hidden_dim, hidden_dim),
                               nn.Tanh()
        )
        self.n_iters = n_iters
        
        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.edgenetwork = nn.Sequential(nn.Linear(2*(hidden_dim+input_dim),2*hidden_dim),
                                         nn.ReLU())

        self.outputnet = nn.Sequential(nn.Linear(hidden_dim+input_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, output_dim),
                                       nn.Sigmoid())

        self.nodenetwork = EdgeConv(nn=convnn,aggr=aggr)

    def forward(self, data):
        #X = data.x
        X = self.batchnorm(data.x)
        H = self.inputnet(X)
        data.x = torch.cat([H,X],dim=-1)
        for i in range(self.n_iters):
            H = self.nodenetwork(data.x,data.edge_index)
            data.x = torch.cat([H,X],dim=-1)
        x_sum = global_mean_pool(data.x, data.batch)
        #row,col = data.edge_index        
        #output = self.edgenetwork(torch.cat([data.x[row],data.x[col]],dim=-1)).squeeze(-1)
        output = self.outputnet(x_sum)
        return output

def dense(nfeatures, ncsc, ncscfeatures, nlabels):
    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv1D, SpatialDropout1D, GRU, Concatenate, Flatten

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
    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv1D, SpatialDropout1D, GRU, Concatenate, Flatten

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

def conv1d_gru(ncsc, ncscfeatures, nlabels):
    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv1D, SpatialDropout1D, GRU, Concatenate, Flatten

    inputs_csc = Input(shape=(ncsc,ncscfeatures,), name='input_csc')
    x = BatchNormalization(name='bn_1')(inputs_csc)
    x = Conv1D(filters=64, kernel_size=(5,), strides=(1,), padding='same', use_bias=False, name='conv1d_1', activation = 'relu')(x)
    x = SpatialDropout1D(rate=0.1)(x)
    x = Conv1D(filters=64, kernel_size=(5,), strides=(1,), padding='same', use_bias=False, name='conv1d_2', activation = 'relu')(x)
    x = SpatialDropout1D(rate=0.1)(x)
    x = GRU(150,go_backwards=True,implementation=2,name='gru_1')(x)
    x = Dropout(rate=0.1)(x)

    outputs = Dense(nlabels, name = 'output', activation='sigmoid')(x)
    keras_model = Model(inputs=inputs_csc, outputs=outputs)

    return keras_model
