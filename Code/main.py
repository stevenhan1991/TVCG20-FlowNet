 import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import argparse
from model import *
from dataio import *
from train import *


parser = argparse.ArgumentParser(description='PyTorch Implementation of FlowNet')
parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--x', type=int, default=51, metavar='N',
                    help='x dimension')
parser.add_argument('--y', type=int, default=51, metavar='N',
                    help='y dimension')
parser.add_argument('--z', type=int, default=51, metavar='N',
                    help='z dimension')
parser.add_argument('--data_type', type=str, default='.dat', metavar='N',
                    help='data type of storing objects')
parser.add_argument('--checkpoint', type=int, default=10, metavar='N',
                    help='checkpoint to save the model (default: 10)')
parser.add_argument('--train_data_path', type=str, default='../train/', metavar='N',
                    help='training data path')
parser.add_argument('--infer_data_path', type=str, default='../inf/', metavar='N',
                    help='inference data path')
parser.add_argument('--model_path', type=str, default='../model/', metavar='N',
                    help='model saving path')
parser.add_argument('--mode', type=str, default='train', metavar='N',
                    help='train or infer the model')

opts = parser.parse_args()
opts.cuda = not opts.no_cuda and torch.cuda.is_available()


def main(opts):
	if opts.mode == 'train':
		train(opts)
	elif opts.mode == 'inf':
		inf(opts)

if __name__== "__main__":
    main()
