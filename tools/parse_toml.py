#!/usr/bin/env python3

import toml
import argparse

parser = argparse.ArgumentParser(description='Generate initial condition.')
parser.add_argument('parameter',
                    help='Path to a parameter file.')
parser.add_argument('get', choices=['seed', 'number', 'kappa'])

args = parser.parse_args()

with open(args.parameter) as conffile:
    config = toml.loads(conffile.read())

if args.get == 'seed':
    print(config['simulation']['seed'])
elif args.get == 'number':
    print(config['simulation']['number_of_particles'])
elif args.get == 'kappa':
    b = config['parameters']['magnetic_reorientation']
    dr = config['parameters']['diffusion']['rotational']
    print(b/dr)
