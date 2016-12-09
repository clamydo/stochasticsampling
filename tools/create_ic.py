#!/usr/bin/env python3

import sys
import json
import cbor2
import toml
import random
import argparse


def error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

parser = argparse.ArgumentParser(description='Generate initial condition.')
parser.add_argument('parameter',
                    help='Path to a parameter file.')

args = parser.parse_args()

with open(args.parameter) as conffile:
    config = toml.loads(conffile.read())

random.seed(config['simulation']['seed'][1])
angles = json.loads(sys.stdin.read())

n = config['simulation']['number_of_particles']

if len(angles) < n:
    error("Number of angles provided ({}) are less then the number of particles require ({})".format(
        len(angles), n))
    sys.exit(1)

box_size = config['simulation']['box_size']

output = [
    {
        'orientation': a,
        'position': {
            'x': random.uniform(0., box_size[0]),
            'y': random.uniform(0., box_size[1])
        }
    }
    for a in angles[:n]
]

cbor2.dump(output, sys.stdout.buffer)

sys.exit(0)
