#!/bin/bash

#SBATCH --job-name=Packet_based_RePO_Adversarial
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH -o job.log
#SBATCH --output=packet_adversarial_2500/Packet_based_RePO_Adversarial_out2.txt
#SBATCH --error=packet_adversarial_2500/Packet_based_RePO_Adversarial_error2.txt

#carregar versão python
module load Python/3.8

#ativar ambiente
source $HOME/RePO/bin/activate

#executar .py
python Packet_based_RePO_Adversarial.py

