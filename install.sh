conda create --name llp-muon-ml python=3.6
conda activate llp-muon-ml
conda install uproot -c conda-forge
conda install pandas tensorflow-gpu keras matplotlib pytables 
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install --verbose --no-cache-dir torch-scatter
pip install --verbose --no-cache-dir torch-sparse
pip install --verbose --no-cache-dir torch-cluster
pip install --verbose --no-cache-dir torch-spline-conv (optional)
pip install torch-geometric
