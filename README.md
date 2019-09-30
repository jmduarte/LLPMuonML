# LLPMuonML

Install (assuming miniconda3 installed):
```
source install.sh
```
or directly using yaml:
```
conda env create -f environment.yml
```

Convert ROOT files to H5:
```
python convert-uproot.py
```

Train Dense/Conv1D/GRU NN:
```
python train_dnn.py
```

Train Graph NN
```
python train_gnn.py
```

