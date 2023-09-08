
# We have got to this whilst thinking about fact that delta and V in above solutions
# look as though they are marching on in pretty perfect sine waves, then what
# happens when they get to future conformal singularity?

# We think this means both Phi and Phidot have to go to zero at end of range (again
# see below), so would be good to have analytic expression for what Phi at
# a=infinity is.

# Can hopefully get this from integral form.

##############
# Flat Universe
##############

PsiFlatInt := a-> a^2/(a^4+1)^(1/2)/(1+a^2*k^2+a^4);
# int(PsiFlatInt(a) ,a=0..infinity);
ArgPsiIntFlat := k -> sqrt((k^2+2)/(k^2-2)) * k * ( EllipticK(1/sqrt(2)) - EllipticPi(1/2-k^2/4, 1/sqrt(2)) );
ArgPsiFlat := k -> 2 * k * exp(-1/4*I*Pi)*(EllipticPi(exp(1/4*I*Pi)*a, 1/2*I*(k^2 - sqrt(k^4 - 4)), I) - EllipticPi(exp(1/4*I*Pi)*a, 1/2*I*(k^2 + sqrt(k^4 - 4)), I));

##############
# Curved Universe
##############

PsiCurvedInt := a-> a^2/ sqrt(a^4+1-2*kc*a^2) / (1+a^2*(k^2+6*kc)+a^4);

kc := -1.2:
a := 1:
ArgPsiCurved := k-> 2 * sqrt(k^2+8*kc) * sqrt(kc- sqrt(kc^2-1)) * ( EllipticPi( a/ sqrt(kc- sqrt(kc^2-1)) , - 1/2* (kc- sqrt(kc^2-1)) *( (k^2+6*kc) - sqrt((k^2+6*kc)^2 - 4)), sqrt(kc^2-1) -kc ) - EllipticPi( a/ sqrt(kc- sqrt(kc^2-1)) , - 1/2* (kc- sqrt(kc^2-1)) *( (k^2+6*kc) + sqrt((k^2+6*kc)^2 - 4)), sqrt(kc^2-1) -kc ));
plot (ArgPsiCurved(k), k = 0..6);
