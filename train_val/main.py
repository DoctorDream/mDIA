import torch.cuda

from trainer import Trainer
from config import Config
from evaluator import Evaluator
from dialogpt_evaluator import DialogptEvaluator
from parameters import parse
import os
import torch
batch_size = 16



def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model_name = args.model_name
    config = Config(args.lang, args.batch_size)
    config.args_dict['learning_rate'] = args.learning_rate
    config.args_dict['warmup_steps'] = args.warmup_steps
    config.args_dict['output_dir'] += f'_{args.learning_rate}_{args.warmup_steps}'
    trainer = Trainer(model_name, config, args.translated)
    trainer.train(small=args.small)



def evaluate(args):
    torch.cuda.set_device(int(args.gpu))
    model_name = args.model_name
    config = Config(args.lang, args.batch_size, args.strategy)
    config.metric = args.metric
    if 'mt5' in model_name:
        evaluator = Evaluator(model_name, config)
    elif 'DialoGPT' in model_name:
        evaluator = DialogptEvaluator(model_name, config, args.translated)

    evaluator.evaluate(small=args.small)




def main(args):
    if args.do == 'train':
        train(args)
    elif args.do == 'eval':
        evaluate(args)


if __name__ == '__main__':
    args = parse()
    main(args)
