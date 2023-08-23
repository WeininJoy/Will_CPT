BeginPackage["targetFunctions`"]
PhiFlat::usage = "PhiFlat[a, K]";
Begin["Private`"]
PhiFlat[{a_, K_}] := N[Re[(1 + a^4)^(1/4) Exp[1/2 I ArcTan[a^2]] HeunG[-1, 1/4 (5 - K^2 I), 1, 5/2, 5/2, 1/2, a^2 I]]];
End[]
EndPackage[]