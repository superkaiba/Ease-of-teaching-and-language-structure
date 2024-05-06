"""
Evaluating teaching speed of language during resets

"""
from __future__ import print_function
from __future__ import division
import torch
import parser
import os
import random
import sys
import pdb 
import wandb

from metrics_saver import Saver
args = parser.parse()  # parsed argument from CLI
run = wandb.init(entity="yoshua-bengio", dir=args['wandbdir'], group=args['wandbgroup'], config=args, project="ease_of_teaching")

if not os.path.exists(args['fname']):
    os.makedirs(args['fname'])

print(args)

from dataGenerator import Dataset
import utility
import numpy as np
if args['population']:
  from popgame import popGuessGame
else:
  from game import GuessGame

torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
np.random.seed(args['seed'])
random.seed(args['seed'])

torch.backends.cudnn.deterministic=True
# @title Train
team = GuessGame(args)

# get data
data = Dataset(args)
util = utility.Utility(args, data)

metrics_saver = Saver(args['fname'], ['i', 'loss', 'train_accuracy', 'topo_measure'])

# with torch.no_grad():
#     topo_measure, d_entropy, sentences, corresponding_objects = util.get_sender_language(team, neural=True)  # evaluate all group performance
# setfacl -Rdm user:vedant.shah:rwx $SCRATCH/causal_imputation/data
# setfacl -Rm  user:vedant.shah:rwx $SCRATCH/causal_imputation/data
# setfacl -m   user:vedant.shah:x   $SCRATCH/causal_imputation
# setfacl -m   user:vedant.shah:x   $SCRATCH

for i in range(args['trainIters']):
    candidates, targets, targets_idx = data.getBatchData(None, args['batchSize'], args['distractNum'])

    loss, train_accuracy, message, predicted = team.forward(targets, targets_idx, candidates, False, True, True, stochastic=True)
    team.backward(loss)
    # print intermediate results during training
    if i % 100 == 0:
        record = 'Iteration ' + str(i) \
                + ' Loss ' + str(np.round(loss.item(), decimals=4)) \
                + ' Training accuracy ' + str(np.round(train_accuracy * 100, decimals=2)) + '%\n'
        print(record)
        wandb.log({'i': i, 'loss': loss.item(), 'train_accuracy': train_accuracy}, step=i)

    if i % args['topo_eval_interval'] == 0:
        with torch.no_grad():
            topo_measure, d_entropy, sentences, corresponding_objects = util.get_sender_language(team, neural=True, topo_n_samples=10000) # calculate topo similarity before each reset
        torch.save(sentences, args['fname'] + f'/i={i}_topo_measure={topo_measure}_accuracy={train_accuracy}_w.pt')
        torch.save(corresponding_objects, args['fname'] + f'/i={i}_topo_measure={topo_measure}_accuracy={train_accuracy}_z.pt')
        # np.save(args['fname'] + f'/i={i}_topo_measure={topo_measure}_accuracy={train_accuracy}_w.npy', sentences.cpu())
        # np.save(args['fname'] + f'/i={i}_topo_measure={topo_measure}_accuracy={train_accuracy}_z.npy', corresponding_objects)
        wandb.log({'i': i, 'topo_measure': topo_measure}, step=i)

    if i != 0 and i % args['resetIter'] == 0 and args['reset']:
        print('Resetting receiver')
        print('Currently on receiver #' + str(i // args['resetIter']))
        team.resetReceiver()
    
    if i % args['saveInterval'] == 0:
        metrics = {
            "i": i,
            "loss":loss,
            "train_accuracy": train_accuracy, 
            # "entropy": entropy, 
            # "listener_entropy": listener_entropy, 
            "topo_measure": topo_measure, 
            # "d_entropy": d_entropy
            }
        metrics_saver.save(metrics)

print('After training for ' + str(args['trainIters']) + ' iterations')
with torch.no_grad():
    topo_measure, d_entropy, sentences, corresponding_objects = util.get_sender_language(team, neural=True) # evaluate all group performance

np.save(args['fname'] + '/w.npy', sentences.cpu())
np.save(args['fname'] + '/z.npy', corresponding_objects)

metrics = {
        "i": i,
        "loss": loss,
        # "sender_loss": sloss, 
        # "receiver_loss": rloss, 
        "train_accuracy": train_accuracy, 
        # "entropy": entropy, 
        # "listener_entropy": listener_entropy, 
        "topo_measure": topo_measure, 
        # "d_entropy": d_entropy
        }
metrics_saver.save(metrics)

torch.save(team.sbot, args['fname'] + '/sbot')
torch.save(team.rbot, args['fname'] + '/rbot')