#!/bin/bash

MYPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARAMETER_FILE="$1"
RANDOM_SCRIPT="$2"

FULLSEED=$($MYPATH/parse_toml.py "$PARAMETER_FILE" seed)

SEED=$(expr "$FULLSEED" : '^\[\([0-9]*\), [0-9]*]')
KAPPA=$($MYPATH/parse_toml.py "$PARAMETER_FILE" kappa)
N=$($MYPATH/parse_toml.py "$PARAMETER_FILE" number)

INITPATH="initial"
mkdir $INITPATH &> /dev/null

COORDS="$INITPATH/coords-seed$SEED-kappa$KAPPA-N$N.json"
INITCON="$INITPATH/initial-seed$SEED-kappa$KAPPA-N$N.cbor"


echo -n "Sampling coordinates from distribution..."

/usr/bin/env wolframscript -script "$MYPATH/$RANDOM_SCRIPT" $SEED $KAPPA $N > "$COORDS"

echo "done"


echo -n "Creating initial condition file..."

$MYPATH/create_ic.py "$PARAMETER_FILE" < "$COORDS" > "$INITCON"

echo "done -> $INITCON"
