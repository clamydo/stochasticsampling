#!/usr/bin/env wolframscript -script

f=Exp[\[Kappa] Cos[\[Phi]]];
norm = Integrate[f,{\[Phi],0,2\[Pi]}];
\[Psi][\[Phi]_,\[Kappa]_]=f/norm;


pd[\[Kappa]_]:=ProbabilityDistribution[\[Psi][\[Phi],\[Kappa]],{\[Phi],0,2 \[Pi]}]


seed=ToExpression[$ScriptCommandLine[[2]]]
\[Kappa]=ToExpression[$ScriptCommandLine[[3]]]
n=ToExpression[$ScriptCommandLine[[4]]]
SeedRandom[seed]
Print@ExportString[RandomVariate[pd[\[Kappa]], n],"JSON"]
