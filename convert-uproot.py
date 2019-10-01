import uproot
import pandas
import numpy as np
import pandas as pd
import h5py
import tables
import sys
filters = tables.Filters(complevel=7, complib='blosc')

path_sig = '/mnt/hadoop/store/group/phys_exotica/delayedjets/displacedJetMuonAnalyzer/V1p7/MC_Summer16/v11/v5/signals/wH/normalized/'
path_bkg = '/mnt/hadoop/store/group/phys_exotica/delayedjets/displacedJetMuonAnalyzer/V1p7/MC_Summer16/v11/v5/bkg/wH/normalized/'
infiles = [
    path_sig+'WH_HToSSTobbbb_WToLNu_MH-125_MS-15_ctauS-10000_TuneCUETP8M1_13TeV-powheg-pythia8_1pb_weighted.root',
    path_sig+'WH_HToSSTobbbb_WToLNu_MH-125_MS-15_ctauS-1000_TuneCUETP8M1_13TeV-powheg-pythia8_1pb_weighted.root',
    path_sig+'WH_HToSSTobbbb_WToLNu_MH-125_MS-15_ctauS-100_TuneCUETP8M1_13TeV-powheg-pythia8_1pb_weighted.root',
    path_sig+'WH_HToSSTobbbb_WToLNu_MH-125_MS-40_ctauS-10000_TuneCUETP8M1_13TeV-powheg-pythia8_1pb_weighted.root',
    path_sig+'WH_HToSSTobbbb_WToLNu_MH-125_MS-40_ctauS-1000_TuneCUETP8M1_13TeV-powheg-pythia8_1pb_weighted.root',
    path_sig+'WH_HToSSTobbbb_WToLNu_MH-125_MS-40_ctauS-100_TuneCUETP8M1_13TeV-powheg-pythia8_1pb_weighted.root',
    path_sig+'WH_HToSSTobbbb_WToLNu_MH-125_MS-55_ctauS-10000_TuneCUETP8M1_13TeV-powheg-pythia8_1pb_weighted.root',
    path_sig+'WH_HToSSTobbbb_WToLNu_MH-125_MS-55_ctauS-1000_TuneCUETP8M1_13TeV-powheg-pythia8_1pb_weighted.root',
    path_sig+'WH_HToSSTobbbb_WToLNu_MH-125_MS-55_ctauS-100_TuneCUETP8M1_13TeV-powheg-pythia8_1pb_weighted.root',
    path_bkg+'WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_1pb_weighted.root'
]

outfile = 'data/raw/WH_HToSSTobbbb_WToLNu_MH-125_MS-15to55_ctauS-100to10000_WJetsToLNu.h5'
entrystop = 10000
other_branches = ['runNum', 'lumiSec', 'evtNum', 'npv', 'rho', 'met', 'metPhi', 'nCsc', 
                  'nCscClusters', 'nCscITClusters', 'nLeptons', 'nJets',
                  'gLLP_eta','gLLP_decay_vertex_r','gLLP_decay_vertex_z']
csc_branches = ['cscITLabels', 'cscLabels', 'cscStation', 'cscLayer',
                'cscPhi', 'cscEta', 'cscX', 'cscY', 'cscZ', 
                'cscDirectionX', 'cscDirectionY', 'cscDirectionZ',
                'cscNRecHits', 'cscNRecHits_flag', 'cscT', 'cscChi2']
cluster_branches = ['cscClusterX','cscClusterY','cscClusterZ','cscClusterTime',
                    'cscClusterTimeSpread','cscClusterTimeRMS','cscClusterGenMuonDeltaR',
                    'cscClusterRadius','cscClusterMajorAxis','cscClusterMinorAxis',
                    'cscClusterEtaPhiSpread','cscClusterPhiSpread','cscClusterEtaSpread',
                    'cscClusterXSpread','cscClusterYSpread','cscClusterZSpread',
                    'cscClusterPhi','cscClusterEta','cscClusterJetVeto',
                    'cscClusterMuonVeto','cscClusterCaloJetVeto','cscClusterSize',
                    'cscClusterMe1112Ratio','cscClusterNStation','cscClusterMaxStation',
                    'cscClusterMaxStationRatio','cscClusterNLayer','cscClusterMaxLayer',
                    'cscClusterMaxLayerRatio','cscClusterVertexR','cscClusterVertexZ',
                    'cscClusterVertexDis','cscClusterVertexChi2','cscClusterVertexN1',
                    'cscClusterVertexN5','cscClusterVertexN10','cscClusterVertexN15',
                    'cscClusterVertexN20','cscClusterVertexN']
lepton_branches = ['lepPt','lepEta','lepPhi']
jet_branches = ['jetPt','jetEta','jetPhi']

print(other_branches)
print(csc_branches)
print(cluster_branches)
print(lepton_branches)
print(jet_branches)

def _write_carray(a, h5file, name, group_path='/', **kwargs):
    h5file.create_carray(group_path, name, obj=a, filters=filters, createparents=True, **kwargs)
    
def _transform(dataframe, max_particles=100, start=0, stop=-1):    
    return dataframe[dataframe.index.get_level_values(-1)<max_particles].unstack().fillna(0)

df_others = []
df_cscs = []
df_clusters = []
df_leptons = []
df_jets = []
currententry = 0
for infile in infiles:
    upfile = uproot.open(infile)
    tree = upfile['MuonSystem']
    
    df_other = tree.pandas.df(branches=other_branches, entrystart=0, entrystop = entrystop)
    if 'HToSSTobbbb_WToLNu' in infile:
        df_other['isSignal'] = np.ones(len(df_other))
        mask_llp0 = (np.abs(df_other['gLLP_decay_vertex_z[0]']) > 568) & (np.abs(df_other['gLLP_decay_vertex_z[0]']) < 1100) & (np.abs(df_other['gLLP_eta[0]']) > 0.9) & (np.abs(df_other['gLLP_eta[0]']) < 2.4) & (df_other['gLLP_decay_vertex_r[0]'] < 695.5)
        mask_llp1 = (np.abs(df_other['gLLP_decay_vertex_z[1]']) > 568) & (np.abs(df_other['gLLP_decay_vertex_z[1]']) < 1100) & (np.abs(df_other['gLLP_eta[1]']) > 0.9) & (np.abs(df_other['gLLP_eta[1]']) < 2.4) & (df_other['gLLP_decay_vertex_r[1]'] < 695.5)
        mask_csc = (df_other['nCsc'] > 30)
        mask = (mask_llp0 | mask_llp1) & (mask_csc)
    else:
        df_other['isSignal'] = np.zeros(len(df_other))
        mask = (df_other['nCsc'] > 30)
    df_other_original = df_other
    df_other = df_other[mask]
    print(df_other)
    df_csc = tree.pandas.df(branches=csc_branches, entrystart=0, entrystop = entrystop)
    df_cluster = tree.pandas.df(branches=cluster_branches, entrystart=0, entrystop = entrystop)
    df_lepton = tree.pandas.df(branches=lepton_branches, entrystart=0, entrystop = entrystop)
    df_jet = tree.pandas.df(branches=jet_branches, entrystart=0, entrystop = entrystop)

    df_other.index = df_other.index+currententry
    df_csc.index = df_csc.index.set_levels(df_csc.index.levels[0]+currententry, level=0)
    df_cluster.index = df_cluster.index.set_levels(df_cluster.index.levels[0]+currententry, level=0)
    df_lepton.index = df_lepton.index.set_levels(df_lepton.index.levels[0]+currententry, level=0)
    df_jet.index = df_jet.index.set_levels(df_jet.index.levels[0]+currententry, level=0)
    currententry += len(df_other_original)

    df_others.append(df_other)
    df_cscs.append(df_csc)
    df_clusters.append(df_cluster)
    df_leptons.append(df_lepton)
    df_jets.append(df_jet)
    
df_other = pd.concat(df_others)
df_csc = pd.concat(df_cscs)
df_cluster = pd.concat(df_clusters)
df_lepton = pd.concat(df_leptons)
df_jet = pd.concat(df_jets)

print(df_other)
# shuffle
df_other = df_other.sample(frac=1)
# apply new ordering to other dataframes
df_csc = df_csc.reindex(df_other.index.values,level=0)
df_cluster = df_cluster.reindex(df_other.index.values,level=0)
df_lepton = df_lepton.reindex(df_other.index.values,level=0)
df_jet = df_jet.reindex(df_other.index.values,level=0)
print(df_other)
with tables.open_file(outfile, mode='w') as h5file:
    #max_csc = len(df_csc.index.get_level_values(-1).unique())
    max_csc = 136
    max_cluster = len(df_cluster.index.get_level_values(-1).unique())
    max_lepton = len(df_lepton.index.get_level_values(-1).unique())
    max_jet = len(df_jet.index.get_level_values(-1).unique())

    print("max_csc",max_csc)
    print("max_lepton",max_lepton)
    print("max_jet",max_jet)
    
    v_csc = _transform(df_csc, max_particles = max_csc)
    for k in csc_branches:
        v = np.stack([v_csc[(k, i)].values for i in range(max_csc)], axis=-1)
        _write_carray(v, h5file, name=k)

    v_cluster = _transform(df_cluster, max_particles = max_cluster)
    for k in cluster_branches:
        v = np.stack([v_cluster[(k, i)].values for i in range(max_cluster)], axis=-1)
        _write_carray(v, h5file, name=k)
                
    v_lepton = _transform(df_lepton, max_particles = max_lepton)
    for k in lepton_branches:
        v = np.stack([v_lepton[(k, i)].values for i in range(max_lepton)], axis=-1)
        _write_carray(v, h5file, name=k)

    v_jet = _transform(df_jet, max_particles = max_jet)
    for k in jet_branches:
        v = np.stack([v_jet[(k, i)].values for i in range(max_jet)], axis=-1)
        _write_carray(v, h5file, name=k)
        
    for k in df_other.columns:
        _write_carray(df_other[k].values, h5file, name=k.replace('[','').replace(']',''))

f = tables.open_file(outfile)
print(f)
f.close()
