#!/bin/bash

#SBATCH --job-name=tfbmn
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH -o job.log
#SBATCH --output=train_flow_new/train_flow_based_model_new_out.txt
#SBATCH --error=train_flow_new/train_flow_based_model_new_error.txt

#carregar versão python
module load Python/3.8

#ativar ambiente
source $HOME/RePO/bin/activate

#executar .py
python train_flow_based_model_new.py

