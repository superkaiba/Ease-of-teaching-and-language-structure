#!/bin/bash

#SBATCH --partition=main
#SBATCH --cpus-per-task=4                             
#SBATCH --gres=gpu:v100:1                                    
#SBATCH --mem=8G                                       
#SBATCH --time=8:00:00                                
#SBATCH -o /network/scratch/t/thomas.jiralerspong/kolmogorov/slurm/slurm-%j.out

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export WANDB_MODE=online

conda init bash
source activate ease_of_teaching
cd /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code


python /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code/main_lstm.py \
    --reset False \
    --expname recreate_original_no_reset \
    --fname $SCRATCH/kolmogorov/ease_of_teaching/results/new \
    --wandbdir $SCRATCH/kolmogorov/ease_of_teaching \
    --wandbgroup ease_of_teaching \
    --n_attributes 2 \
    --n_values 8 \
    --vocabSize 8 \
    --messageLen 2 \
    --hiddenSize 100 \
    --resetIter 6000 \
    --trainIters 300000

python /home/mila/t/thomas.jiralerspong/kolmogorov/Ease-of-teaching-and-language-structure/code/main_lstm.py \
    --reset True \
    --expname recreate_original_with_reset \
    --fname $SCRATCH/kolmogorov/ease_of_teaching/results/new \
    --wandbdir $SCRATCH/kolmogorov/ease_of_teaching \
    --wandbgroup ease_of_teaching \
    --n_attributes 2 \
    --n_values 8 \
    --vocabSize 8 \
    --messageLen 2 \
    --hiddenSize 100 \
    --resetIter 6000 \
    --trainIters 300000