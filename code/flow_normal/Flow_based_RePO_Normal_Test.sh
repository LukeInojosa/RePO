#!/bin/bash

#SBATCH --job-name=FBRNT
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH -o job.log
#SBATCH --output=flow_normal/Flow_based_RePO_Normal_Test_out.txt
#SBATCH --error=flow_normal/Flow_based_RePO_Normal_Test_error.txt

#carregar versão python
module load Python/3.8

#ativar ambiente
source $HOME/RePO/bin/activate

#executar .py
python Flow_based_RePO_Normal_Test.py

