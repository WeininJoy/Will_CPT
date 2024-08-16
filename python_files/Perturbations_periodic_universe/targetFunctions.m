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
  // heun = (I*a^2/alpha - 1)^(-gamma) * (alpha*I*a^2 + 1)^gamma * HeunG[-1/alpha^2, (5*alpha - ks^2*I)/(4*alpha), 1, 5/2, 5/2, 1/2, a^2*I/alpha] ;
  heun = (I*a^2/alpha - 1)^(-gamma) * (alpha*I*a^2 + 1)^gamma * HeunG[-1/alpha^2, (5*alpha - (ks^2-8 kc)*I)/(4*alpha), 1, 5/2, 5/2, 1/2, a^2*I/alpha] ;
  Phi = N[Re[ f * heun]];
  Return[Phi]]
End[]

PsiIntFlat::usage = "PsiIntFlat[{ks}]";
Begin["Private`"]
PsiIntFlat[ks_] := N[ Re[ ((ks^2+2)/(ks^2-2))**0.5 * ks * ( EllipticK[1/2] - EllipticPi[1/2-ks^2/4, 1/2] ) ] ];
End[]

PsiIntCurved::usage = "PsiIntCurved[{a, ks, kc}]";
Begin["Private`"]
// PsiIntCurved[{a_, ks_, kc_}] := N[ Re [(ks^2+8*kc)^0.5 * (kc- (kc^2-1)^0.5)^0.5 * ( EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (ks^2+6*kc) - ((ks^2+6*kc)^2 - 4)^0.5), ArcSin[a/ (kc- (kc^2-1)^0.5 )^0.5] , ((kc^2-1)^0.5 -kc)^2 ] - EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (ks^2+6*kc) + ((ks^2+6*kc)^2 - 4)^0.5), ArcSin [a/ (kc- (kc^2-1)^0.5)^0.5] , ((kc^2-1)^0.5 -kc)^2 ] ) ]];
PsiIntCurved[{a_, ks_, kc_}] := N[ Re [(ks^2)^0.5 * (kc- (kc^2-1)^0.5)^0.5 * ( EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (ks^2-2*kc) - ((ks^2-2*kc)^2 - 4)^0.5), ArcSin[a/ (kc- (kc^2-1)^0.5 )^0.5] , ((kc^2-1)^0.5 -kc)^2 ] - EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (ks^2-2*kc) + ((ks^2-2*kc)^2 - 4)^0.5), ArcSin [a/ (kc- (kc^2-1)^0.5)^0.5] , ((kc^2-1)^0.5 -kc)^2 ] ) ]];
End[]

PhiIntCurved::usage = "PhiIntCurved[{a, ks, kc}]";
Begin["Private`"]
PhiIntCurved[{a_, ks_, kc_}] :=
 Block[{f, psi, Phi}, 
  // f = 3* (a^4 + (ks^2+6*kc)*a^2 +1)^0.5 / (ks^2 + 8*kc)^0.5 / ((ks^2 + 6*kc)^2 - 4)^0.5 / a^3;
  // psi = (kc- (kc^2-1)^0.5)^0.5 * ( EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (ks^2+6*kc) - ((ks^2+6*kc)^2 - 4)^0.5), ArcSin[a/ (kc- (kc^2-1)^0.5 )^0.5] , ((kc^2-1)^0.5 -kc)^2 ] - EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (ks^2+6*kc) + ((ks^2+6*kc)^2 - 4)^0.5), ArcSin [a/ (kc- (kc^2-1)^0.5)^0.5] , ((kc^2-1)^0.5 -kc)^2 ] );
  // Phi = N[ Re[ f * Sin[(ks^2 + 8*kc)^0.5 * psi] ]];
  f = 3* (a^4 + (ks^2-2*kc)*a^2 +1)^0.5 / (ks^2)^0.5 / ((ks^2 -2*kc)^2 - 4)^0.5 / a^3;
  psi = (kc- (kc^2-1)^0.5)^0.5 * ( EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (ks^2-2*kc) - ((ks^2-2*kc)^2 - 4)^0.5), ArcSin[a/ (kc- (kc^2-1)^0.5 )^0.5] , ((kc^2-1)^0.5 -kc)^2 ] - EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (ks^2-2*kc) + ((ks^2-2*kc)^2 - 4)^0.5), ArcSin [a/ (kc- (kc^2-1)^0.5)^0.5] , ((kc^2-1)^0.5 -kc)^2 ] );
  Phi = N[ Re[ f * Sin[(ks^2)^0.5 * psi] ]];
  Return[Phi]]
End[]

dPhidkCurved::usage = "dPhidkCurved[{a, ks, kc}]";
Begin["Private`"]
dPhidkCurved[{a_, ks_, kc_}] :=
 Block[{dPhidkR},
  // f[k_] := 3* (a^4 + (k^2+6*kc)*a^2 +1)^0.5 / (k^2 + 8*kc)^0.5 / ((k^2 + 6*kc)^2 - 4)^0.5 / a^3;
  // psi[k_] := (kc- (kc^2-1)^0.5)^0.5 * ( EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (k^2+6*kc) - ((k^2+6*kc)^2 - 4)^0.5), ArcSin[a/ (kc- (kc^2-1)^0.5 )^0.5] , ((kc^2-1)^0.5 -kc)^2 ] - EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (k^2+6*kc) + ((k^2+6*kc)^2 - 4)^0.5), ArcSin [a/ (kc- (kc^2-1)^0.5)^0.5] , ((kc^2-1)^0.5 -kc)^2 ] );
  // Phi[k_] := f[k] * Sin[(k^2 + 8*kc)^0.5 * psi[k]];
  f[k_] := 3* (a^4 + (k^2-2*kc)*a^2 +1)^0.5 / (k^2 + 8*kc)^0.5 / ((k^2 -2*kc)^2 - 4)^0.5 / a^3;
  psi[k_] := (kc- (kc^2-1)^0.5)^0.5 * ( EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (k^2-2*kc) - ((k^2-2*kc)^2 - 4)^0.5), ArcSin[a/ (kc- (kc^2-1)^0.5 )^0.5] , ((kc^2-1)^0.5 -kc)^2 ] - EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (k^2-2*kc) + ((k^2-2*kc)^2 - 4)^0.5), ArcSin [a/ (kc- (kc^2-1)^0.5)^0.5] , ((kc^2-1)^0.5 -kc)^2 ] );
  Phi[k_] := f[k] * Sin[(k^2)^0.5 * psi[k]];
  dPhidk[k_] := Evaluate@D[Phi[k], {k, 1}];
  dPhidkR = N[Re[ dPhidk[ks]]];
  Return[dPhidkR]]
End[]

d2PhidkCurved::usage = "d2PhidkCurved[{a, ks, kc}]";
Begin["Private`"]
d2PhidkCurved[{a_, ks_, kc_}] :=
 Block[{d2PhidkR},
  // f[k_] := 3* (a^4 + (k^2+6*kc)*a^2 +1)^0.5 / (k^2 + 8*kc)^0.5 / ((k^2 + 6*kc)^2 - 4)^0.5 / a^3;
  // psi[k_] := (kc- (kc^2-1)^0.5)^0.5 * ( EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (k^2+6*kc) - ((k^2+6*kc)^2 - 4)^0.5), ArcSin[a/ (kc- (kc^2-1)^0.5 )^0.5] , ((kc^2-1)^0.5 -kc)^2 ] - EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (k^2+6*kc) + ((k^2+6*kc)^2 - 4)^0.5), ArcSin [a/ (kc- (kc^2-1)^0.5)^0.5] , ((kc^2-1)^0.5 -kc)^2 ] );
  // Phi[k_] := f[k] * Sin[(k^2 + 8*kc)^0.5 * psi[k]];
  f[k_] := 3* (a^4 + (k^2-2*kc)*a^2 +1)^0.5 / (k^2 )^0.5 / ((k^2 -2*kc)^2 - 4)^0.5 / a^3;
  psi[k_] := (kc- (kc^2-1)^0.5)^0.5 * ( EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (k^2-2*kc) - ((k^2-2*kc)^2 - 4)^0.5), ArcSin[a/ (kc- (kc^2-1)^0.5 )^0.5] , ((kc^2-1)^0.5 -kc)^2 ] - EllipticPi[ - 1/2* (kc- (kc^2-1)^0.5) *( (k^2-2*kc) + ((k^2-2*kc)^2 - 4)^0.5), ArcSin [a/ (kc- (kc^2-1)^0.5)^0.5] , ((kc^2-1)^0.5 -kc)^2 ] );
  Phi[k_] := f[k] * Sin[(k^2 )^0.5 * psi[k]];
  d2Phidk[k_] := Evaluate@D[Phi[k], {k, 2}];
  d2PhidkR = N[Re[ d2Phidk[ks]]];
  Return[d2PhidkR]]
End[]

EndPackage[]