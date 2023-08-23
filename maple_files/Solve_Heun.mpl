# odeFlat := diff(w(z), z$2) + (9*z^2 - 2*z - 5)/(2*z*(z - 1)*(z + 1))* diff(w(z), z$1) + (10*z - 5 + k^2*I)/(4*z*(z - 1)*(z + 1))*w(z) = 0 ;
# dsolve(odeFlat);

odeCurved := diff(w(z), z$2) + (9/2*z^2 - (3*alpha - 3/alpha +1) *z/alpha - 5/(2*alpha^2) ) / ( z* (z-1)* (z+1/alpha^2) ) * diff(w(z), z$1) + ( (5/2*z + (I*k^2/4 - 5/4)/alpha) / (z*(z-1)*(z+1/alpha^2)) - (1/4*(alpha-1/alpha)/alpha^2) / ((z-1)*(z+1/alpha^2))^2 )* w(z) = 0 ;
sol := dsolve(odeCurved);

# odeCurvedtest := diff(w(z), z$2) + (-9/2*z^2 + (6*K*I+1)*z + 5/2) / (z * (1-z^2+2*K*I*z)) * diff(w(z), z$1) + ( (-5/2*z + (5/4-I*k^2/4)) / (z* (1-z^2+2*K*I*z)) + (-1/2*K*I) / (1-z^2+2*K*I*z)^2 ) * w(z) = 0 ;
# dsolve(odeCurvedtest);

beta := (alpha-1)*alpha / 2;
sol_curved := w(z) = (z-1)^(2*alpha^(-beta) +2) * (alpha^2*z +1)^(2*alpha^beta +2) * HeunG(-1/alpha^2, (k^2*I-5*alpha) /(4*alpha), 1, 5/2, 5/2, 1/2, z);
odetest(sol, odeCurved);