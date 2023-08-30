BeginPackage["targetFunctions`"]

PhiFlat::usage = "PhiFlat[{a, ks}]";
Begin["Private`"]
PhiFlat[{a_, ks_}] := N[Re[(1 + a^4)^(1/4) Exp[1/2 I ArcTan[a^2]] HeunG[-1, 1/4 (5 - ks^2 I), 1, 5/2, 5/2, 1/2, a^2 I]]];
End[]

PhiCurved::usage = "PhiCurved[{a, ks, kc}]";
Begin["Private`"]
PhiCurved[{a_, ks_, kc_}] := 
 Block[{alpha, gamma, f, heun, Phi}, 
  alpha = I * kc + ( 1 - kc^2)^(1/2);
  gamma = alpha * (alpha - 1) / (2* alpha^2 + 2);
  f = Exp[-1/2 * (1-I*kc)* ArcTan[(-I*kc + I*a^2) / (kc^2-1)^0.5] / (kc^2-1)^0.5] * Exp[I/4 * ArcTan[2*kc*I*a^2 / (a^4+1)]] * (1 + (2-4*kc^2)*a^4 + a^8)^(1/8) ;
  heun = (I*a^2/alpha - 1)^(-gamma) * (alpha*I*a^2 + 1)^gamma * HeunG[-1/alpha^2, (5*alpha - ks^2*I)/(4*alpha), 1, 5/2, 5/2, 1/2, a^2*I/alpha] ;
  Phi = N[Re[ f * heun]];
  Return[Phi]]
End[]

EndPackage[]