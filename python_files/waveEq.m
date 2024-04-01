BeginPackage["waveEq`"]

NumSol::usage = "NumSol[{kappa}]";
Begin["Private`"]
NumSol[{kappa_}] :=
Block[{pde, ufun}, 
    pde = D[u[t, x], {t, 2}] - D[u[t, x], {x, 2}] + kappa^2 u[t, x] == 0;
    ufun = NDSolveValue[{pde, u[t, 0] == 0, u[t, 1] == 0, u[0, x] == 1, u[1, x] == 0}, u, {t, 0, 1}, {x, 0, 1}];
    Return[ufun]]
End[]

EndPackage[]