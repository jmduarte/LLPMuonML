import keras
import numpy as np
import tables
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def get_raw_features_labels(file_name, features, features_lep, features_jet, labels):
    # load file
    h5file = tables.open_file(file_name, 'r')
    nevents = getattr(h5file.root,features[0]).shape[0]
    nlep = getattr(h5file.root,features_lep[0]).shape[1]
    njet = getattr(h5file.root,features_jet[0]).shape[1]
    nlabels = len(labels)
    nfeatures = len(features)
    nfeatures_lep = len(features_lep)
    nfeatures_jet = len(features_jet)
    # allocate arrays
    feature_array = np.zeros((nevents,nfeatures))
    feature_lep_array = np.zeros((nevents,nlep,nfeatures_lep))
    feature_jet_array = np.zeros((nevents,njet,nfeatures_jet))
    label_array = np.zeros((nevents,nlabels))
    # load feature arrays
    for (i, feat) in enumerate(features):
        feature_array[:,i] = getattr(h5file.root,feat)[:]
    # load lep feature arrays
    for (i, feat) in enumerate(features_lep):
        feature_lep_array[:,:,i] = getattr(h5file.root,feat)[:,:]
    # load jet feature arrays
    for (i, feat) in enumerate(features_jet):
        feature_jet_array[:,:,i] = getattr(h5file.root,feat)[:,:]
    # load labels arrays
    for (i, label) in enumerate(labels):
        label_array[:,i] = getattr(h5file.root,label)[:]

    h5file.close()
    return feature_array, feature_lep_array, feature_jet_array, label_array


file_path = 'data/raw/WH_HToSSTobbbb_WToLNu_MH-125_MS-40_ctauS-10000_WJetsToLNu.h5'
features = ['met', 'metPhi', 'nCsc',
            'nCscClusters', 'nCscITClusters', 
            'nLeptons', 'nJets']
features_lep = ['lepPt','lepEta','lepPhi']
features_jet = ['jetPt','jetEta','jetPhi']
labels = ['isSignal']
feature_array, feature_lep_array, feature_jet_array, label_array = get_raw_features_labels(file_path,features,features_lep,features_jet,labels)
nevents = label_array.shape[0]
nfeatures = feature_array.shape[1]
nlabels = len(labels)
print(feature_array.shape)
print(feature_lep_array.shape)
print(feature_jet_array.shape)

inputs = Input(shape=(nfeatures,), name = 'input')  
x = BatchNormalization(name='bn_1')(inputs)
x = Dense(64, name = 'dense_1', activation='relu')(x)
x = Dense(32, name = 'dense_2', activation='relu')(x)
x = Dense(32, name = 'dense_3', activation='relu')(x)
outputs = Dense(nlabels, name = 'output', activation='softmax')(x)
keras_model = Model(inputs=inputs, outputs=outputs)
keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(keras_model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
callbacks = [early_stopping, model_checkpoint]

# fit keras model
keras_model.fit(feature_array, label_array, batch_size=1024, 
                epochs=100, validation_split=0.2, shuffle=False,
                callbacks = callbacks)
