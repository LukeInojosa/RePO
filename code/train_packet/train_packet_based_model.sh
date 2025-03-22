#!/bin/bash

#SBATCH --job-name=RePO_train_packet_based_model
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem 32G
#SBATCH -o job.log
#SBATCH --output=train_packet_based_model_out.txt
#SBATCH--error=train_packet_based_model_error.txt

#carregar versão python
module load Python/3.8

#criar ambiente
python -m venv $HOME/RePO

#ativar ambiente
source $HOME/RePO/bin/activate

#instalar pacotes desejados
pip install --upgrade pip
pip install tensorflow==2.3.0
pip install pandas==1.0.3
pip install numpy==1.18.1
pip install --upgrade protobuf==3.20.3
   
#executar .py
python train_packet_based_model.py
