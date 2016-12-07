# How to use?
First generate orientation angles with the Mathematica script

    /usr/bin/env wolframscript -script spathomdist_init.wl {seed} {kappa} {n} > out.angles

Where you have to insert the `{seed}`, the value of `{kappa}` and the number of
particles `{n}`. This script outputs orientation angles sampled from a
distribtion as a JSON array.

This array can now be read with `create_ic.py`, which has to be called like

    ./create_ic.py {seed} {box size x} {box size y} < out.angles > inital_condition.cbor

`initial_condition.cbor` now includes a CBOR serialized array of a dictonary
with the field `position` and `orientation`.

# Troubleshooting
## Configuring wolframscript
Configuring wolframscript can be necessary. If you get an error like

    Use -configure to set WOLFRAMSCRIPT_KERNELPATH
    Or export WolframKernel=/yourpath/WolframKernel

try to set the WolframKernel path with

    wolframscript -configure WOLFRAMSCRIPT_KERNELPATH=/path/to/Wolfram/Mathematica/11.0/Executables/wolfram
