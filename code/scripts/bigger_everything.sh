#!/bin/bash

#SBATCH --partition=long,main,unkillable,lab-bengioy
#SBATCH --cpus-per-task=4                             
#SBATCH --gres=gpu:v100:1                                    
#SBATCH --mem=8G                                       
#SBATCH --time=4:00:00                                
#SBATCH -o /network/scratch/t/thomas.jiralerspong/kolmogorov/slurm/slurm-%j.out

# conda init bash
# source activate ease_of_teaching
cd /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code
python /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code/main.py \
    --reset False \
    --fname /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code/logs/noreset50 \
    --n_attributes 5 \
    --n_values 10 \
    --vocabSize 20 \
    --messageLen 10 \
    --hiddenSize 512
