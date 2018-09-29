import argparse
from yaml import load
from io import open
from toolz.dicttoolz import merge
from runner import Runner

parser = argparse.ArgumentParser(description='Bezos')
parser.add_argument('--config', default='test.yaml',
                    help='Configuration file')
args = parser.parse_args()


def main():
    stream = open(args.config, 'r')
    default = open('./configs/default.yaml', 'r')
    parameters = load(stream)
    default_parameters = load(default)
    parameters = merge(default_parameters, parameters)
    runner = Runner(**parameters)
    runner.run()


if __name__ == "__main__":
    main()
