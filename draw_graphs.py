import setGPU
import torch
from torch_geometric.data import Data, DataLoader
from graph_data import GraphDataset
from models import EdgeNet
import os
import os.path as osp
import math
import numpy as np
import tqdm
import argparse
import matplotlib.pyplot as plt
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

def main(args): 

    full_dataset = GraphDataset(root='data/', connect_all=True)
    #full_dataset = GraphDataset(root='data_stations/', connect_all=False)
    
    data = full_dataset.get(30)
    print('isSignal', data.y)
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
    df = pd.DataFrame(data.x.cpu().detach().numpy()*feature_scale, columns=features)
    df['cscR'] = np.sqrt(df['cscX']*df['cscX']+df['cscY']*df['cscY']+df['cscZ']*df['cscZ'])
    mask = np.abs(df['cscX'])>0
    df = df[mask]
    row, col = data.edge_index.cpu().detach().numpy()    

    for x,y in [('cscX', 'cscY'), ('cscZ', 'cscY'), ('cscEta', 'cscPhi'), ('cscZ','cscT'), ('cscR', 'cscEta'), ('cscZ','cscR')]:
        plt.figure()       
        for i, j in tqdm.tqdm(zip(row, col)):
            if i > j: continue
            seg_args = dict(c='b',alpha=0.3,zorder=1)
            plt.plot([df[x][i], df[x][j]],
                     [df[y][i], df[y][j]], '-', **seg_args)
        plt.scatter(df[x], df[y],c='k',marker='o',s=4,zorder=2,alpha=0.7)
        plt.xlabel("%s"%x)
        plt.ylabel("%s"%y)
        plt.figtext(0.25, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
        plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
        plt.savefig('graph_%s_%s.pdf'%(x,y))

    from mpl_toolkits import mplot3d
    for x,y,z in [('cscX', 'cscY', 'cscZ')]:
        fig = plt.figure()  
        ax = plt.axes(projection='3d')     
        for i, j in tqdm.tqdm(zip(row, col)):
            if i > j: continue
            seg_args = dict(c='b',alpha=0.3,zorder=1)
            ax.plot3D([df[x][i], df[x][j]],
                      [df[y][i], df[y][j]],
                      [df[z][i], df[z][j]], '-', **seg_args)
        ax.scatter3D(df[x], df[y], df[z], c='k',marker='o',s=4,zorder=2,alpha=0.7)
        ax.set_xlabel("%s"%x)
        ax.set_ylabel("%s"%y)
        ax.set_zlabel("%s"%z)
        plt.figtext(0.25, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
        plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
        plt.savefig('graph_%s_%s_%s.pdf'%(x,y,z))
        for ii in range(0,360,1):
            ax.view_init(elev=10., azim=ii)
            plt.savefig("movie%03d.png" % ii)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
