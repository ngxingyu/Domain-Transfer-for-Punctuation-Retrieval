# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local #~
# source ~/.bashrc
# conda create --name NLP python=3.8
conda activate NLP

pip uninstall torchtext
pip install snoop nemo_toolkit[all]==1.0.0b2 transformers datasets hydra-core git+https://github.com/pabloppp/pytorch-tools -U
# conda install -c conda-forge pytorch-lightning=1.1.5
pip install pytorch-lightning==1.1.5
# Set cudatoolkit version based on gpu version.
conda install pytorch cudatoolkit=11.2 -c pytorch #10.1
