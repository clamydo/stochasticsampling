#!/usr/bin/env python3

import sys
import json
import cbor
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
coords = json.loads(sys.stdin.read())

n = config['simulation']['number_of_particles']

if len(coords) < n:
    error("Number of coords provided ({}) are less then the number of particles require ({})".format(
        len(coords), n))
    sys.exit(1)

box_size = config['simulation']['box_size']

output = [
    {
        'orientation': float(a[2]),
        'position': {
            'x': float(a[0] * box_size[0]),
            'y': float(a[1] * box_size[1])
        }
    }
    for a in coords[:n]
]

cbor.dump(output, sys.stdout.buffer)

sys.exit(0)
