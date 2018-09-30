import argparse
from yaml import load
from utils import print_dic
from io import open
from toolz.dicttoolz import merge
from runner import Runner
from evaluator import Evaluator

parser = argparse.ArgumentParser(description='Bezos')
parser.add_argument('--config', default='test.yaml',
                    help='Configuration file')
parser.add_argument('command', choices=['train', 'evaluate'], default='train')
parser.add_argument('--load-dir', action='store', dest='load_dir',
                    help='Trained model dir', default="./trained_models/")
parser.add_argument('--det', action='store_true',
                    dest='det', help='Deterministic evaluation')
args = parser.parse_args()

header = """
▀█████████▄     ▄████████  ▄███████▄   ▄██████▄     ▄████████ 
  ███    ███   ███    ███ ██▀     ▄██ ███    ███   ███    ███ 
  ███    ███   ███    █▀        ▄███▀ ███    ███   ███    █▀  
 ▄███▄▄▄██▀   ▄███▄▄▄      ▀█▀▄███▀▄▄ ███    ███   ███        
▀▀███▀▀▀██▄  ▀▀███▀▀▀       ▄███▀   ▀ ███    ███ ▀███████████ 
  ███    ██▄   ███    █▄  ▄███▀       ███    ███          ███ 
  ███    ███   ███    ███ ███▄     ▄█ ███    ███    ▄█    ███ 
▄█████████▀    ██████████  ▀████████▀  ▀██████▀   ▄████████▀  
"""


def main():
    print(header)
    stream = open(args.config, 'r')
    default = open('./configs/default.yaml', 'r')
    parameters = load(stream)
    default_parameters = load(default)
    if(args.command == 'train'):
        parameters = merge(default_parameters, parameters)
        print("Training parameters\n-------")
        print_dic(parameters)
        runner = Runner(**parameters)
        runner.run()
    else:
        parameters = merge(merge(default_parameters, parameters), {
            'deterministic_evaluation': args.det,
            'load_dir': args.load_dir
        })
        evaluator = Evaluator(**parameters)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
