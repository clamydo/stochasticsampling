#!/usr/bin/env python3

from sys import stdin, stdout
import json
import cbor2
import random
import argparse

parser = argparse.ArgumentParser(description='Generate initial condition.')
parser.add_argument('seed', type=int,
                    help='Seed for pseudo random number generator.')
parser.add_argument('x', type=float,
                    help='Box size in x-direction.')
parser.add_argument('y', type=float,
                    help='Box size in y-dreictin.')

args = parser.parse_args()

random.seed(args.seed)
angles = json.loads(stdin.read())

output = [
    {
        'orientation': a,
        'position': {
            'x': random.uniform(0., args.x),
            'y': random.uniform(0., args.y)
        }
    }
    for a in angles
]

cbor2.dump(output, stdout.buffer)

# with open('output.cbor', 'wb') as fp:
#     cbor2.dump(output, fp)
