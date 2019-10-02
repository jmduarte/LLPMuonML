import setGPU
import keras
import numpy as np
import tables
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
from models import dense, dense_conv1d_gru, conv1d_gru
import math

def get_raw_features_labels(file_name, features, features_csc, features_lep, features_jet, labels, features_csc_scale):
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
    #feature_jet_array = np.zeros((nevents,njet,nfeatures_jet))
    label_array = np.zeros((nevents,nlabels))
    # load feature arrays
    for (i, feat) in enumerate(features):
        feature_array[:,i] = getattr(h5file.root,feat)[:]
    # load csc feature arrays
    for (i, feat) in enumerate(features_csc):
        feature_csc_array[:,:,i] = getattr(h5file.root,feat)[:,:]/features_csc_scale[i]
    # load lep feature arrays
    for (i, feat) in enumerate(features_lep):
        feature_lep_array[:,:,i] = getattr(h5file.root,feat)[:,:]
    # load jet feature arrays
    #for (i, feat) in enumerate(features_jet):
    #    feature_jet_array[:,:,i] = getattr(h5file.root,feat)[:,:]
    # load labels arrays
    for (i, label) in enumerate(labels):
        label_array[:,i] = getattr(h5file.root,label)[:]

    h5file.close()
    return feature_array, feature_csc_array, feature_lep_array, label_array

def main(args):
    file_path = 'data/raw/WH_HToSSTobbbb_WToLNu_MH-125_MS-15to55_ctauS-100to10000_WJetsToLNu.h5'
    features = ['npv', 'rho', 'met', 'metPhi', 'nCsc', 'nCscClusters', 'nCscITClusters', 
                'nLeptons', 'nJets']
    features_csc = ['cscX', 'cscY', 'cscZ', 'cscT', 
                    'cscEta', 'cscPhi', 
                    'cscDirectionX', 'cscDirectionY', 'cscDirectionZ',
                    'cscStation', 'cscLayer', 
                    'cscNRecHits', 'cscChi2']
    features_csc_scale = np.array([1000., 1000., 1000., 100., 
                                   3., np.pi, 
                                   1., 1., 1.,
                                   40., 4.,
                                   6. , 1000.])
    features_lep = ['lepPt','lepEta','lepPhi']
    features_jet = ['jetPt','jetEta','jetPhi']
    labels = ['isSignal']
    feature_array, feature_csc_array, feature_lep_array, label_array = get_raw_features_labels(file_path,features,features_csc,features_lep,features_jet,labels,features_csc_scale)
    nevents = label_array.shape[0]
    nfeatures = feature_array.shape[1]
    ncsc = feature_csc_array.shape[1]
    ncscfeatures = feature_csc_array.shape[2]
    nlabels = len(labels)

    keras_model = conv1d_gru(ncsc,ncscfeatures,nlabels)

    keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(keras_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    # fit keras model
    X = feature_csc_array
    y = label_array

    fulllen = len(y)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    w_s = 1.
    w_b = (np.sum(y)/len(y)) / (np.sum(1-y)/len(y))
    weights = (y)*(w_s)+(1-y)*(w_b)
    weights = weights.reshape(-1)

    feature_array_train = feature_array[0:splits[0]]
    feature_array_val = feature_array[splits[1]:splits[2]]
    feature_array_test = feature_array[splits[0]:splits[1]]

    feature_csc_array_train = feature_csc_array[0:splits[0]]
    feature_csc_array_val = feature_csc_array[splits[1]:splits[2]]
    feature_csc_array_test = feature_csc_array[splits[0]:splits[1]]

    weights_train = weights[0:splits[0]]
    weights_val = weights[splits[1]:splits[2]]
    weights_test = weights[splits[0]:splits[1]]

    y_train = y[0:splits[0]]
    y_val = y[splits[1]:splits[2]]
    y_test = y[splits[0]:splits[1]]
    

    X_train = feature_csc_array_train
    X_val = feature_csc_array_val
    X_test = feature_csc_array_test
    
    keras_model.fit(X_train, y_train, sample_weight = weights_train, batch_size=1024, 
                    epochs=100, validation_data=(X_val, y_val, weights_val), shuffle=True,
                    callbacks = callbacks)

    keras_model.load_weights('keras_model_best.h5')
    
    predict_test = keras_model.predict(X_test)

    fpr = {}
    tpr = {}
    auc1 = {}
    
    plt.figure()       
    fpr, tpr, threshold = roc_curve(y_test,predict_test)
    auc_test = auc(fpr, tpr)
    
    plt.plot(tpr,fpr,label='LLP muon tagger (Conv1D+GRU+Dense), AUC = %.1f%%'%(auc_test*100.))
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
    plt.savefig('ROC_dnn.pdf')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
