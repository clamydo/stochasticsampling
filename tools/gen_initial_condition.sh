#!/bin/bash

MYPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARAMETER_FILE=$1

FULLSEED=$($MYPATH/parse_toml.py "$PARAMETER_FILE" seed)

SEED=$(expr "$FULLSEED" : '^\[\([0-9]*\), [0-9]*]')
KAPPA=$($MYPATH/parse_toml.py "$PARAMETER_FILE" kappa)
N=$($MYPATH/parse_toml.py "$PARAMETER_FILE" number)

INITPATH="initial"
mkdir $INITPATH &> /dev/null

ANGLES="$INITPATH/angless-seed$SEED-kappa$KAPPA-N$N.json"
INITCON="$INITPATH/initial-seed$SEED-kappa$KAPPA-N$N.cbor"


echo -n "Sampling angles from distribution..."

/usr/bin/env wolframscript -script "$MYPATH/spathomdist_init.wl" $SEED $KAPPA $N > "$ANGLES"

echo "done"


echo -n "Sampling angles from distribution..."

$MYPATH/create_ic.py "$PARAMETER_FILE" < "$ANGLES" > "$INITCON"

echo "done"
