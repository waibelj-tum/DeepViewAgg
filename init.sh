pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchnet
CUDA=cu116

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
pip install torch-geometric


python -c "import torch;from torch_geometric.data import Data"

pip install torch-points-kernels==0.6.10 --no-cache-dir

pip install omegaconf
pip install wandb
pip install tensorboard
pip install plyfile
pip install hydra-core==1.1.0
pip install pytorch-metric-learning
pip install matplotlib
pip install seaborn
pip install pykeops==1.4.2
pip install imageio
pip install opencv-python
pip install pypng
pip install git+http://github.com/CSAILVision/semantic-segmentation-pytorch.git@master
pip install h5py
pip install faiss-gpu

#sudo apt install libopenblas-dev
export CUDA_HOME=/usr/local/cuda-11.6
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

python -c "from torch import nn; import MinkowskiEngine as ME"

#sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
python -c "from torch import nn; from torchsparse import nn as spnn"
pip install pykeops==1.4.1

pip install plotly==5.4.0
pip install "jupyterlab>=3" "ipywidgets>=7.6"
pip install jupyter-dash
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.1

PATH="$HOME/anaconda3/envs/deep_view_aggregation2/lib/python3.7/site-packages/torch/lib"
ln -s $PATH/libnvrtc-builtins.so $PATH/libnvrtc-builtins.so.11.1
