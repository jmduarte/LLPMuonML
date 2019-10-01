import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools
import tables
import numpy as np

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['WH_HToSSTobbbb_WToLNu_MH-125_MS-15to55_ctauS-100to10000_WJetsToLNu.h5']

    @property
    def processed_file_names(self):
        nevents = 0
        for raw_file_name in self.raw_file_names:
            h5file = tables.open_file(osp.join(self.raw_dir, raw_file_name),'r')
            nevents += h5file.root.nCsc.shape[0]
            h5file.close()
        return ['data_{}.pt'.format(i) for i in range(nevents)]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def get_raw_features_labels(self, file_name, features, spectators, labels):
        # load file
        h5file = tables.open_file(file_name, 'r')
        nevents = getattr(h5file.root,features[0]).shape[0]
        ncsc = getattr(h5file.root,features[1]).shape[1]
        nlabels = len(labels)
        nspectators = len(spectators)
        nfeatures = len(features)
        # allocate arrays
        feature_array = np.zeros((nevents,ncsc,nfeatures))
        spec_array = np.zeros((nevents,nspectators))
        label_array = np.zeros((nevents,nlabels))
        # load feature arrays
        for (i, feat) in enumerate(features):
            feature_array[:,:,i] = getattr(h5file.root,feat)[:,:]
        # load spectator arrays
        for (i, spec) in enumerate(spectators):
            spec_array[:,i] = getattr(h5file.root,spec)[:]
        # load labels arrays
        for (i, label) in enumerate(labels):
            label_array[:,i] = getattr(h5file.root,label)[:]

        h5file.close()
        return feature_array, spec_array, label_array

    def process(self):
        i = 0    
        features = ['cscX', 'cscY', 'cscZ', 'cscT', 
                    'cscEta', 'cscPhi', 
                    'cscDirectionX', 'cscDirectionY', 'cscDirectionZ',
                    'cscStation', 'cscLayer', 
                    'cscNRecHits', 'cscChi2']
        feature_scale = np.array([1000., 1000., 1000., 100., 
                                  3., np.pi, 
                                  1., 1., 1.,
                                  40., 4.,
                                  6. , 1000.])
        spectators = ['nCsc']
        labels = ['isSignal']
        for raw_path in self.raw_paths:
            feature_array, spec_array, label_array = self.get_raw_features_labels(raw_path,features,spectators,labels)
            nevents = label_array.shape[0]
            for event in range(nevents):
                max_csc = min(spec_array[event,0],feature_array.shape[1])
                pairs = [[i, j] for (i, j) in itertools.product(range(int(max_csc)),range(int(max_csc))) if i!=j]
                edge_index = torch.tensor(pairs, dtype=torch.long)
                edge_index=edge_index.t().contiguous()
                x = torch.tensor(feature_array[event]/feature_scale, dtype=torch.float)
                y = torch.tensor(label_array[event], dtype=torch.float).reshape(1,1)
                data = Data(x=x, edge_index=edge_index, y=y)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


