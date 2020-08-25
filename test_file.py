import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--maze_seed_list", default='5')

print(parser.parse_args().maze_seed_list.split(','))