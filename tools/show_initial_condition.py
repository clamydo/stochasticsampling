#!/usr/bin/env python3
import cbor2
import sys

obj = cbor2.load(sys.stdin.buffer)
print(obj)
