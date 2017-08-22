#!/usr/bin/env python3

import cbor
import msgpack
import toml
import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser(
    description='Reconstruct parameter file from metadata')
parser.add_argument('--out', help='Defaults to "parameter_reconstructed.toml"',
                    default='parameter_reconstructed.toml')
parser.add_argument('--format', help='file format of simulation output',
                    default='MsgPack')
parser.add_argument('--index', help='Path to index file')
parser.add_argument('data', help='Path to simulation data', type=str)

args = parser.parse_args()


sim_setting = {}

index = np.fromfile(args.index, dtype=np.uint64)

with open(args.data, 'rb') as f:
    if args.format == 'MsgPack':
        buf = f.read(index[0])
        sim_settings = msgpack.unpackb(buf, encoding='utf-8')
    elif args.format == 'CBOR':
        sim_settings = cbor.load(f).value
    else:
        print('ERROR: Unsupported format: {}'.format(args.format))
        sys.exit(1)


print(sim_settings)

with open(args.out, 'w') as f:
    toml.dump(sim_settings, f)
