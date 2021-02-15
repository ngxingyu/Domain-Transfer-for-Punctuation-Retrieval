wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -f -p ~
source ~/.bashrc
conda create --name NLP python=3.8
conda activate NLP
pip install snoop nemo_toolkit[all]==1.0.0b2 transformers datasets hydra-core git+https://github.com/pabloppp/pytorch-tools -U
conda install -c conda-forge pytorch-lightning
conda install pytorch cudatoolkit=10.1 -c pytorch
