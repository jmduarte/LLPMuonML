import uproot
import pandas
import numpy as np
import pandas as pd
import h5py
import tables
import sys
filters = tables.Filters(complevel=7, complib='blosc')

infiles = ['/mnt/hadoop/store/group/phys_exotica/delayedjets/displacedJetMuonAnalyzer/V1p7/MC_Summer16/v11/v5/signals/wH/normalized/WH_HToSSTobbbb_WToLNu_MH-125_MS-40_ctauS-10000_TuneCUETP8M1_13TeV-powheg-pythia8_1pb_weighted.root',
          '/mnt/hadoop/store/group/phys_exotica/delayedjets/displacedJetMuonAnalyzer/V1p7/MC_Summer16/v11/v5/bkg/wH/normalized/WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_1pb_weighted.root']

outfile = 'data/raw/WH_HToSSTobbbb_WToLNu_MH-125_MS-40_ctauS-10000_WJetsToLNu.h5'
entrystop = 1500 #take the first 1500 events in each file
#entrystop = None
other_branches = ['runNum', 'lumiSec', 'evtNum', 'npv', 'rho', 'met', 'metPhi', 'nCsc', 
                  'nCscClusters', 'nCscITClusters', 'nLeptons', 'nJets']
csc_branches = ['cscITLabels', 'cscLabels', 'cscStation', 'cscLayer',
                'cscPhi', 'cscEta', 'cscX', 'cscY', 'cscZ', 
                'cscDirectionX', 'cscDirectionY', 'cscDirectionZ',
                'cscNRecHits', 'cscNRecHits_flag', 'cscT', 'cscChi2']
lepton_branches = ['lepPt','lepEta','lepPhi']
jet_branches = ['jetPt','jetEta','jetPhi']

print(other_branches)
print(csc_branches)
print(lepton_branches)
print(jet_branches)

def _write_carray(a, h5file, name, group_path='/', **kwargs):
    h5file.create_carray(group_path, name, obj=a, filters=filters, createparents=True, **kwargs)
    
def _transform(dataframe, max_particles=100, start=0, stop=-1):    
    return dataframe[dataframe.index.get_level_values(-1)<max_particles].unstack().fillna(0)

df_others = []
df_cscs = []
df_leptons = []
df_jets = []
currententry = 0
for infile in infiles:
    upfile = uproot.open(infile)
    tree = upfile['MuonSystem']

    df_other = tree.pandas.df(branches=other_branches, entrystart=0, entrystop = entrystop)
    if 'WH_HToSSTobbbb_WToLNu' in infile:
        df_other['isSignal'] = np.ones(len(df_other))
    else:
        df_other['isSignal'] = np.zeros(len(df_other))
    df_csc = tree.pandas.df(branches=csc_branches, entrystart=0, entrystop = entrystop)
    df_lepton = tree.pandas.df(branches=lepton_branches, entrystart=0, entrystop = entrystop)
    df_jet = tree.pandas.df(branches=jet_branches, entrystart=0, entrystop = entrystop)

    df_other.index = df_other.index+currententry
    df_csc.index = df_csc.index.set_levels(df_csc.index.levels[0]+currententry, level=0)
    df_lepton.index = df_lepton.index.set_levels(df_lepton.index.levels[0]+currententry, level=0)
    df_jet.index = df_jet.index.set_levels(df_jet.index.levels[0]+currententry, level=0)
    currententry += len(df_other)

    df_others.append(df_other)
    df_cscs.append(df_csc)
    df_leptons.append(df_lepton)
    df_jets.append(df_jet)
    
df_other = pd.concat(df_others)
df_csc = pd.concat(df_cscs)
df_lepton = pd.concat(df_leptons)
df_jet = pd.concat(df_jets)

# shuffle
df_other = df_other.sample(frac=1)
# apply new ordering to other dataframes
df_csc = df_csc.reindex(df_other.index.values,level=0)
df_lepton = df_lepton.reindex(df_other.index.values,level=0)
df_jet = df_jet.reindex(df_other.index.values,level=0)

with tables.open_file(outfile, mode='w') as h5file:
    
    max_csc = len(df_csc.index.get_level_values(-1).unique())
    max_lepton = len(df_lepton.index.get_level_values(-1).unique())
    max_jet = len(df_jet.index.get_level_values(-1).unique())

    print("max_csc",max_csc)
    print("max_lepton",max_lepton)
    print("max_jet",max_jet)
    
    v_csc = _transform(df_csc, max_particles = max_csc)
    for k in csc_branches:
        v = np.stack([v_csc[(k, i)].values for i in range(max_csc)], axis=-1)
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
        _write_carray(df_other[k].values, h5file, name=k)

f = tables.open_file(outfile)
print(f)
f.close()
