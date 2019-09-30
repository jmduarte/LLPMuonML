import setGPU
import keras
import numpy as np
import tables
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
from models import dense, dense_conv1d_gru
import os
print(os.environ['CUDA_VISIBLE_DEVICES'])

def get_raw_features_labels(file_name, features, features_csc, features_lep, features_jet, labels):
    # load file
    h5file = tables.open_file(file_name, 'r')
    nevents = getattr(h5file.root,features[0]).shape[0]
    ncsc = getattr(h5file.root,features_csc[0]).shape[1]
    nlep = getattr(h5file.root,features_lep[0]).shape[1]
    njet = getattr(h5file.root,features_jet[0]).shape[1]
    nlabels = len(labels)
    nfeatures = len(features)
    nfeatures_csc = len(features_csc)
    nfeatures_lep = len(features_lep)
    nfeatures_jet = len(features_jet)
    # allocate arrays
    feature_array = np.zeros((nevents,nfeatures))
    feature_csc_array = np.zeros((nevents,ncsc,nfeatures_csc))
    feature_lep_array = np.zeros((nevents,nlep,nfeatures_lep))
    feature_jet_array = np.zeros((nevents,njet,nfeatures_jet))
    label_array = np.zeros((nevents,nlabels))
    # load feature arrays
    for (i, feat) in enumerate(features):
        feature_array[:,i] = getattr(h5file.root,feat)[:]
    # load csc feature arrays
    for (i, feat) in enumerate(features_csc):
        feature_csc_array[:,:,i] = getattr(h5file.root,feat)[:,:]
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
    return feature_array, feature_csc_array, feature_lep_array, feature_jet_array, label_array

def main(args):
    file_path = 'data/raw/WH_HToSSTobbbb_WToLNu_MH-125_MS-40_ctauS-10000_WJetsToLNu.h5'
    features = ['npv', 'rho', 'met', 'metPhi', 'nCsc', 'nCscClusters', 'nCscITClusters', 
                'nLeptons', 'nJets']
    features_csc = ['cscStation', 'cscLayer', 'cscPhi', 'cscEta', 
                    'cscX', 'cscY', 'cscZ', 'cscT', 
                    'cscDirectionX', 'cscDirectionY', 'cscDirectionZ',
                    'cscNRecHits', 'cscChi2']
    features_lep = ['lepPt','lepEta','lepPhi']
    features_jet = ['jetPt','jetEta','jetPhi']
    labels = ['isSignal']
    feature_array, feature_csc_array, feature_lep_array, feature_jet_array, label_array = get_raw_features_labels(file_path,features,features_csc,features_lep,features_jet,labels)
    nevents = label_array.shape[0]
    nfeatures = feature_array.shape[1]
    ncsc = feature_csc_array.shape[1]
    ncscfeatures = feature_csc_array.shape[2]
    nlabels = len(labels)

    #keras_model = dense(nfeatures,ncsc,ncscfeatures,nlabels)
    keras_model = dense_conv1d_gru(nfeatures,ncsc,ncscfeatures,nlabels)

    keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(keras_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    # fit keras model
    X = [feature_array, feature_csc_array]
    y = label_array
    feature_array_train_val, feature_array_test, feature_array_csc_train_val, feature_array_csc_test, y_train_val, y_test = train_test_split(feature_array, feature_csc_array, y, test_size=0.1, random_state=42)
    X_train_val = [feature_array_train_val, feature_array_csc_train_val]
    X_test = [feature_array_test, feature_array_csc_test]
    
    keras_model.fit(X_train_val, y_train_val, batch_size=1024, 
                    epochs=100, validation_split=0.111, shuffle=False,
                    callbacks = callbacks)
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
