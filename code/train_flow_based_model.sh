#!/bin/bash

#SBATCH --job-name=flow
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH -o job.log
#SBATCH --output=train_flow_based_model_out.txt
#SBATCH --error=train_flow_based_model_error.txt

#carregar versão python
module load Python/3.8

#ativar ambiente
source $HOME/RePO/bin/activate

#executar .py
python train_flow_based_model.py


