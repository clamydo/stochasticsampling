#!/usr/bin/env python3

import cbor
import toml
import argparse

parser = argparse.ArgumentParser(
    description='Reconstruct parameter file from metadata')
parser.add_argument('--out', help='Defaults to "parameter_reconstructed.toml"',
                    default='parameter_reconstructed.toml')
parser.add_argument('data', help='Path to simulation data', type=str)

args = parser.parse_args()

with open(args.data, 'rb') as f:
    sim_settings = cbor.load(f).value

with open(args.out, 'w') as f:
    toml.dump(sim_settings, f)
