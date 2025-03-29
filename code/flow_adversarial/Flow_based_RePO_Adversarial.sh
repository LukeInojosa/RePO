#!/bin/bash

#SBATCH --job-name=FadvN
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH -o job.log
#SBATCH --output=flow_adversarial/Flow_based_RePO_Adversarial_out4.txt
#SBATCH --error=flow_adversarial/Flow_based_RePO_Adversarial_error4.txt

#carregar versão python
module load Python/3.8

#ativar ambiente
source $HOME/RePO/bin/activate

#executar .py
python Flow_based_RePO_Adversarial.py

