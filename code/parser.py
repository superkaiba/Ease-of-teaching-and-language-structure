import argparse
import torch 
from datetime import datetime
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse():
    parser = argparse.ArgumentParser(description='Referential game settings')

    parser.add_argument('--gpu', type=int, default=0, help='which gpu if we use gpu')
    parser.add_argument('--fname', type=str, default='test', help='folder name to save results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--jupyter', action='store_true') 
    parser.add_argument('--slambda', type=float, default=0.1, help='speaker regularization hyperparameter')
    parser.add_argument('--rlambda', type=float, default=0.1, help='listener regularization hyperparameter')
    parser.add_argument('--receiverNum', type=int, default=1, help='number of listeners in the population')
    parser.add_argument('--topk', type=int, default=3, help='number of top messages when we probe language')
    parser.add_argument('--evaluateSize', type=int, default=1000, help='the batch size of test objects when not enumeration')
    parser.add_argument('--reset', type=str2bool, default=False, help='whether to reset the game')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes')
    parser.add_argument('--n_values', type=int, default=5, help='number of values per attribute')
    parser.add_argument('--distractNum', type=int, default=5, help='number of objects at each iteration')
    parser.add_argument('--hiddenSize', type=int, default=100, help='hidden size of the model')
    parser.add_argument('--vocabSize', type=int, default=5, help='vocabulary size')
    parser.add_argument('--messageLen', type=int, default=4, help='length of the message')
    parser.add_argument('--topo_eval_interval', type=int, default=50000, help='interval of topology evaluation')
    parser.add_argument('--batchSize', type=int, default=100, help='batch size')
    parser.add_argument('--trainIters', type=int, default=50000, help='number of training iterations')
    parser.add_argument('--sLearnRate', type=float, default=0.001, help='speaker learning rate')
    parser.add_argument('--rLearnRate', type=float, default=0.001, help='listener learning rate')
    parser.add_argument('--resetNum', type=int, default=50, help='number of resets')
    parser.add_argument('--resetIter', type=int, default=200, help='number of iterations before reset')
    parser.add_argument('--deterResetNums', type=int, default=30, help='number of deterministic resets')
    parser.add_argument('--deterResetIter', type=int, default=1000, help='number of iterations before deterministic reset')
    parser.add_argument('--population', type=str2bool, default=False, help='whether to use population of receivers')
    parser.add_argument('--saveInterval', type=int, default=100, help='interval of saving metrics')
    parser.add_argument('--wandbgroup', type=str, default='default_group', help='wandb group name')
    parser.add_argument('--wandbdir', type=str, default='./wandb', help='wandb group name')
    parser.add_argument('--listenerNumLayers', type=int, default=2, help='number of layers in the listener')
    parser.add_argument('--listenerHiddenSize', type=int, default=100, help='hidden size of the listener')

    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%b_%d_%Y_%H:%M:%S")
    args_dict = vars(parser.parse_args()) # convert python object to dict
    args_dict['device'] = torch.device("cuda:" + str(args_dict['gpu']) if torch.cuda.is_available() else "cpu")
    args_dict['fname'] = f"{args_dict['fname']}/vocab_size_{args_dict['vocabSize']}_message_len_{args_dict['messageLen']}_hidden_size_{args_dict['hiddenSize']}_reset={str(args_dict['reset'])}_resetIter={str(args_dict['resetIter'])}/{datetime_string}"
    
    return args_dict
