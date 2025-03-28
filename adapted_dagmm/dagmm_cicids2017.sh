#!/bin/bash

#SBATCH --job-name=DAGMM
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH -o job.log
#SBATCH --output=DAGMM_out.txt
#SBATCH--error=DAGMM_error.txt


#carregar versão python
module load Python/3.10

#criar ambiente
python -m venv $HOME/DAGMM

#ativar ambiente
source $HOME/DAGMM/bin/activate

#instalar pacotes desejados
pip install --upgrade pip
pip install torch
pip install pandas
pip install numpy
pip install matplotlib
pip install ipython
pip install torchvision
pip install tqdm
pip install scikit-learn
#executar .py
python dagmm_cicids2017.py
