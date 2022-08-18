import argparse
from typing import Union
from clearml import Task
import time


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='google/mt5-base')
    parser.add_argument('--do', type=str, default='train')
    parser.add_argument('--lang', type=str)
    parser.add_argument('--translated', type=bool, default=False)
    parser.add_argument('--raw', type=bool, default=False)
    parser.add_argument('--strategy', type=str, default='sample')
    parser.add_argument('--small', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--metric', type=str, default='all')
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--warmup_steps', type=int, default='156')

    args = parser.parse_args()
    return args


args = parse()


# python main.py --model_name google/mt5-base --do eval --lang zh --translated True --raw True


