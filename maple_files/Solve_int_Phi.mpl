
#####################
# Flat Universe
#####################

odeFlat := a*(1+a^4)* diff(Phi(a), a$2) + 2*(3*a^4+2)* diff(Phi(a), a$1) + a*(4*a^2+k^2) *Phi(a) = 0 ;

# PhiFlatAnthony := a -> - (-1)^(7/8) * (a^2*I+1)^(3/4) / (a^2-I)^(1/4) * HeunG(-1, 1/4*(5-I*k^2), 1, 5/2, 5/2, 1/2, a^2*I);
# PhiFlat := a -> (1+a^4)^(1/4) * exp(1/2*I*arctan(a^2))* HeunG(-1, 1/4*(5-I*k^2), 1, 5/2, 5/2, 1/2, a^2*I);

# PsiFlat := a -> exp(-1/4*I*Pi)*(EllipticPi(exp(1/4*I*Pi)*a, 1/2*I*(k^2 - sqrt(k^4 - 4)), I) - EllipticPi(exp(1/4*I*Pi)*a, 1/2*I*(k^2 + sqrt(k^4 - 4)), I));
# PhiFlatInt := a -> 3* sqrt(1+a^2*k^2+a^4) / (a^3*k*sqrt(k^4-4)) * sin(k*sqrt(k^4-4)* PsiFlat(a));
# tmp1:= a-> exp(Int(a^2/(a^4+1)^(1/2)/(1+a^2*k^2+a^4)*(-(k^2-2)*(k^2+2))^(1/2)*k,a)) *(1+a^2*k^2+a^4)^(1/2)/a^3;
# tmp2:= a-> exp(-Int(a^2/(a^4+1)^(1/2)/(1+a^2*k^2+a^4)*(-(k^2-2)*(k^2+2))^(1/2)*k,a)) *(1+a^2*k^2+a^4)^(1/2)/a^3;

## use odetest() to check whether they are solutions to the ode.
# simplify( odetest(Phi(a) = PhiFlat(a), odeFlat) );
# simplify( odetest(Phi(a) = PhiFlatInt(a), odeFlat) );
# simplify( odetest(Phi(a) = tmp1(a), odeFlat) );
# simplify( odetest(Phi(a) = tmp2(a), odeFlat) );

## make plot to check whether the solutions are the same
# k := 2;
# plot (PhiFlatAnthony(a) - PhiFlat(a), a = 0..2);
# plot (PhiFlat(a) - PhiFlatInt(a), a = 0..2);


## solve unknown g(a) from the ode

sqrt(g(a)) / a^3 * exp( I * Int( f(a)/ sqrt(1+a^4) / g(a) , a) ):
subs(f(a) = a^2 , %):
# subs(g(a) = A*a^4 + B*a^2 + C , %):
subs(g(a) = (1 + a^2* k^2+ a^4) / k / sqrt(k^4-4) , %):
PhiGuessFlat := simplify(%):
subs(Phi(a) = PhiGuessFlat, odeFlat):
odeguessFlat := simplify(%);


#####################
# Curved Universe
#####################

odeCurved := (a*(1+a^4) - 2*a^3*kc) * diff(Phi(a), a$2) + (2*(3*a^4+2) - 10*a^2*kc) * diff(Phi(a), a$1) + a*(4*a^2+k^2) * Phi(a) = 0 :

sqrt(g(a)) / a^3 * exp( I * Int( f(a)/ sqrt(1+a^4-2*kc*a^2) / g(a) , a) ):
subs(f(a) = a^2 , %):
# subs(g(a) = A*a^4 + B*a^2 + C , %):
subs(g(a) = (1 + a^2* (k^2+6*kc) + a^4) / sqrt( (k^2+8*kc) * ((k^2+6*kc)^2-4) ) , %):
PhiGuessCurved := simplify(%):
subs(Phi(a) = PhiGuessCurved, odeCurved):
odeguessCurved := simplify(%);

