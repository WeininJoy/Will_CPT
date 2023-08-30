with(Physics):

# Perturbation extraction
zero_order := expr -> subs(h=0,expr):
first_order := expr -> expand(coeff(taylor(expr,h,2),h)):
elim_using_from := (var,eq,expr) -> expand(subs(var=solve(eq,var),expr)):

# Definitions
subsV:= expr -> expand(subs({D(V)(f(t))=dVdphi(f(t)),D(dVdphi)(f(t))=d2Vdphi2(f(t)),D[1,1](V)(f(t))=d2Vdphi2(f(t)), D(d2Vdphi2)(f(t))=d3Vdphi3(f(t)), D[1,1,1](V)(f(t))=d3Vdphi3(f(t))},expr)):
subsH:= expr -> expand(subs({diff(a(t),t)=H(t)*a(t)},expr)):
fullsubs:= expr -> expand(simplify(subsH(subsH(subsV(expr))),symbolic)):

# Set up a cartesian coordinate system
Coordinates(X=[r,th,ph,t]):

mp:=1: # planck mass is one for now

# metric
ds2_FRW := (1+2*h*phi(X))*dt^2 - a(t)^2*(1-2*h*psi(X))*(dr^2/(1-k*r^2) + r^2*(dth^2 + sin(th)^2*dph^2));
Setup(metric=ds2_FRW):

# stress-energy-tensor
T[mu,nu] = D_[mu](f(t) + h*df(X)) * D_[nu](f(t) + h*df(X)) - ( D_[alpha](f(t) + h*df(X)) * D_[~alpha](f(t) + h*df(X)) - 2*V(f(t) + h*df(X)))* g_[mu, nu]/2:
Define(%):

# Equations for simplifying the view onto spatial derivatives
D2psi_eq := D2psi(X) = diff(psi(X),t$2) +3*H(t)*diff(psi(X),t)- subsH(first_order(convert(SumOverRepeatedIndices(D_[mu](D_[~mu](h*psi(X)))),diff))):
D2df_eq := D2df(X) = diff(df(X),t$2) +3*H(t)*diff(df(X),t)- subsH(first_order(convert(SumOverRepeatedIndices(D_[mu](D_[~mu](h*df(X)))),diff))):
D2R_eq := D2R(X) = diff(R(X),t$2) +3*H(t)*diff(R(X),t)- subsH(first_order(convert(SumOverRepeatedIndices(D_[mu](D_[~mu](h*R(X)))),diff))):


# Zero order Einstein equations
# =============================
# time - time
# -----------
zero_order(T[0,0]/mp^2-Einstein[0,0])=0:
E00 := fullsubs(%);

# conservation
# ------------
zero_order(SumOverRepeatedIndices(D_[mu](T[mu,0]))) = 0:
cons0 := fullsubs(%/diff(f(t),t));

# potential-less equation for eliminating \dot H
# ----------------------------------------------
Ecurve := elim_using_from(dVdphi(f(t)),cons0,fullsubs(diff(E00,t)/H(t)/6));

# First order Einstein equations
# ==============================
# space - time
# ------------
first_order(T[1,~0]/mp^2-Einstein[1,~0]):
fullsubs(%);
dEi0 := int(%,r) = 0;

# time - time
# -----------
first_order(T[0,~0]/mp^2-Einstein[0,~0])=0:
dE00 := fullsubs(%):
elim_using_from(diff(psi(X),r$2), D2psi_eq, %):
expand(simplify(%));

# time conservation
# -----------------
first_order(SumOverRepeatedIndices(D_[mu](T[mu,0]))):
dcons0 := fullsubs(%):
elim_using_from(diff(df(X),r$2), D2df_eq, %):
expand(simplify(%)):
elim_using_from(diff(f(t),t$2),cons0,%):
expand(%/diff(f(t),t))=0;


# Full system of equations
# ------------------------
[dE00, diff(dEi0,t),dEi0,dcons0]:
expand(subs(phi(X)=psi(X),%)):                        # Remove phi for psi
expand(subs(psi(X)=R(X)-H(t)/diff(f(t),t)*df(X),%)):  # Remove psi for R
elim_using_from(diff(df(X),r$2), D2df_eq, %):         # Simplify spatial derivatives
elim_using_from(diff(R(X),r$2), D2R_eq, %):
expand(simplify(%)):
%;

subs([
D2df(X)=Grad^2/a(t)^2*df,
D2R(X)=Grad^2/a(t)^2*R,
R(X)=R,diff(R(X),t)=Rt,diff(R(X),t$2)=Rtt,
df(X)=df,diff(df(X),t)=dft,diff(df(X),t$2)=dftt
],
%);
fullsubs(%);
elim_using_from(d2Vdphi2(f(t)),fullsubs(diff(cons0,t)),%);
elim_using_from(dVdphi(f(t)),cons0,%);
elim_using_from(V(f(t)),E00,%);
elim_using_from(diff(H(t),t),Ecurve,%);
fullsubs(%);

eliminate(%,[dftt,dft,df])[2][1]:
collect(%,[Rtt,Rt,R]);
ans := %;
ans;
#coeff(%,Rtt);
#expand(%/H(t)^2/diff(f(t),t)/a(t)^4/4);
fullsubs(subs(diff(f(t),t)=z(t)*H(t)/a(t),%));
elim_using_from(diff(H(t),t),Ecurve,%);
fullsubs(subs(diff(f(t),t)=z(t)*H(t)/a(t),%));
subs(Grad^2=Del^2-3*k,%);
subs(Grad^4=(Del^2-3*k)^2,%);
collect(%/a(t)^3/z(t)/H(t)^3/4,[Rtt,Rt,R],w->collect(expand(w),Del));
%;
coeff(%,Rtt);
expand(%/diff(f(t),t)/H(t)^2/a(t)^4/4);
%%;

z(t) = a(t) * sqrt(2*eps(t));
diff(log(rhs(%)),t);
fullsubs(2*%);

#-----------------------------------

expand(D2R_eq*a(t)^2);

SumOverRepeatedIndices(D_[1](D_[~1](h*R(X))) + D_[2](D_[~2](h*R(X))) + D_[3](D_[~3](h*R(X))));
D_[~mu](h*R(X));
D_[0](D_[~0](h*R(X)));
first_order(convert(%,diff));
SumOverRepeatedIndices(D_[~mu](D_[mu](h*R(X)))):
fullsubs(first_order(convert(%,diff)));
simplify(%%%-%);

SumOverRepeatedIndices(D_[~1](D_[1](R(X))) + D_[~2](D_[2](R(X))) + D_[~3](D_[3](R(X)))):
fullsubs(zero_order(convert(%,diff)));

SumOverRepeatedIndices(D_[~mu](D_[mu](h*R(X)))):
fullsubs(first_order(convert(%,diff))):
subs([diff(R(X),t)=Rt, diff(R(X),t$2)=Rtt,diff(R(X),r)=Rr, diff(R(X),r$2)=Rrr,diff(R(X),th)=Rth,diff(R(X),th$2)=Rthth,diff(R(X),ph)=Rph,diff(R(X),ph$2)=Rphph],%):
collect(%,[Rt,Rtt,Rr,Rrr,Rth,Rthth,Rph,Rphph], simplify);
subs(k=0,%);
#coeff(%,diff(R(X),t$2));

SumOverRepeatedIndices(D_[mu](D_[~mu](h*R(X)))):
convert(%,diff):
fullsubs(%):
first_order(%);
expand(simplify(%));

diff(R(X),t$2) +3*H(t)*diff(R(X),t) - subsH(first_order(convert(SumOverRepeatedIndices(D_[mu](D_[~mu](h*R(X)))),diff)));


with(VariationalCalculus):

L := a(t)^3 * diff(f(t),t)^2/H(t)^2 * (
(Grad^2+3*k)*R(t)^2/a(t)^2 
+(Grad^2+3*k)/(Grad^2+3*k-k*diff(f(t),t)^2/2/H(t)^2) * (diff(R(t),t) - k * R(t)/H(t)/a(t)^2)^2
):

op(EulerLagrange(L,t,[R(t)]));
convert(%,diff):
fullsubs(%):
elim_using_from(diff(H(t),t),Ecurve,%):
collect(%,[ diff(R(t),t$2), diff(R(t),t), R(t)], simplify);

coeff(%,diff(R(t),t$2)):
%;
simplify(%);
