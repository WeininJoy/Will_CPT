with(DifferentialGeometry):
with(Tensor):
with(Tools):
zero_order := expr -> subs(h=0,expr):
first_order := expr -> expand(coeff(taylor(expr,h,2),h)):
second_order := expr -> expand(coeff(taylor(expr,h,3),h,2)):
elim_using_from := (var,eq,expr) -> expand(subs(var=solve(eq,var),expr)):

subsH:= expr -> expand(subs({diff(a(t),t)=H(t)*a(t)},expr)):
subsV:= expr -> expand(subs({D(V)(f(t))=dVdphi(f(t)),D(dVdphi)(f(t))=d2Vdphi2(f(t)),D[1,1](V)(f(t))=d2Vdphi2(f(t))},expr)):
#subsang:= expr -> expand(subs({x=Pi/4,y=Pi/4,z=Pi/4},expr)):
subsall:=expr -> subsH(subsV(expr)): 

 
#k:=0:
DGsetup([r,th,ph],M):
g := a(t)^2*(1-2*h*R(t,r,th,ph))*evalDG(dr &t dr / (1-k*r*r) + r^2*(dth &t dth + sin(th)^2* dph &t dph)):
dg := diff(a(t)^2*(1-2*h*R(t,r,th,ph)),t)*evalDG(dr &t dr / (1-k*r*r) + r^2*(dth &t dth + sin(th)^2* dph &t dph)):
gg := InverseMetric(g):
C1 := Christoffel(g):
D_ := f -> CovariantDerivative(f,C1):
D2_ := f -> ContractIndices(RaiseLowerIndices(gg,D_(D_(f)),[1]),[[1,2]]):

trig_simp := expr -> expand(subs([cos(th)^2 = 1-sin(th)^2,cos(th)^3 = cos(th)*(1-sin(th)^2),cos(th)^4 = 1-2*sin(th)^2+sin(th)^4],expr)):


zero_order := expr -> subs(h=0,expr):
first_order := expr -> expand(coeff(taylor(expr,h,2),h)):
second_order := expr -> expand(coeff(taylor(expr,h,3),h,2)):


# Show the curvature term
simplify(zero_order(RicciScalar(g)),symbolic):
first_order(RicciScalar(g)):
trig_simp(elim_using_from(diff(R(t,r,th,ph),r$2),D2R=first_order(D2_(h*R(t,r,th,ph))),%)):
combine(%,trig);

# Show the laplacian
TensorInnerProduct(g,D_(h*R(t,r,th,ph)),D_(h*R(t,r,th,ph))):
second_order(%):
DRDR_eq:= DRDR=trig_simp(%);



Ni := D_(h*varpsi(t,r,th,ph));
N := 1+h*alpha(t,r,th,ph);

Eij := expand(1/2*(dg - 2*SymmetrizeIndices(D_(Ni),[1,2],"Symmetric"))):
E := ContractIndices(RaiseLowerIndices(gg,Eij,[1]),[[1,2]]):
EijEij := ContractIndices(evalDG(Eij &t RaiseLowerIndices(gg,Eij,[1,2])),[[1,3],[2,4]]):

#ContractIndices(D_(N^(-1)*(RaiseLowerIndices(gg,Eij,[2]))),[[2,3]]):
#subsall(zero_order(DGinfo(%,"CoefficientList",[dr])[1]))=0;
#subsall(zero_order(DGinfo(%%,"CoefficientList",[dth])[1]))=0;
#subsall(zero_order(DGinfo(%%%,"CoefficientList",[dph])[1]))=0;

ContractIndices(D_(N^(-1)*(RaiseLowerIndices(gg,Eij,[2]) - E * RaiseLowerIndices(gg,g,[2]))),[[2,3]]):
subsall(first_order(DGinfo(%,"CoefficientList",[dph])[1]))=0:
expand(int(lhs(%),ph)/2*H(t))=0:
trig_simp(%):
Ni_eq := expand(simplify(%));

RicciScalar(g) - N^(-2)*(EijEij-E^2) - N^(-2)*diff(f(t),t)^2 - 2*V(f(t)) :
#trig_simp(subsall(zero_order(%)));
trig_simp(subsall(first_order(%))):
elim_using_from(alpha(t,r,th,ph),Ni_eq,%*a(t)^2/H(t)/4/a(t)^2):
N_eq := expand(simplify(%)):
trig_simp(elim_using_from(diff(R(t,r,th,ph),r$2),D2R=first_order(D2_(h*R(t,r,th,ph))),%)):
trig_simp(elim_using_from(diff(varpsi(t,r,th,ph),r$2),D2varpsi=first_order(D2_(h*varpsi(t,r,th,ph))),%)):
expand(simplify(%)); 


int_by_parts := (expr, u, dv, v, du, x) -> expand(expr - coeff(coeff(expr,u),dv)*u*dv - coeff(coeff(expr,u),dv)*v*du - diff(coeff(coeff(expr,u),dv),x)*v*u):
int_by_parts_ := (expr, du, u, ddu, x) -> expand(expr - coeff(expr,du,2)*du^2 - coeff(expr,du,2)*u*ddu - diff(coeff(expr,du,2),x)*u*du):
kinetic_trick := (expr, u, du, x) -> expand(expr - coeff(coeff(expr,u),du)*u*du - diff(coeff(coeff(expr,u),du),x) * u^2/2):

E00 := H(t)^2 = (diff(f(t),t)^2/2 + V(f(t)))/3 -k/a(t)^2:
cons0 := diff(f(t),t$2) + 3*H(t)*diff(f(t),t) + dVdphi(f(t))=0:
curv := diff(H(t),t) = -diff(f(t),t)^2/2 + k/a(t)^2:

c := a(t)^3*r^2*sin(th)/(1-k*r^2)^(1/2);

S:= c*(1-2*h*R(t,r,th,ph))^(3/2)*(
N*RicciScalar(g) 
+N^(-1)*(EijEij-E^2) 
+N^(-1)*diff(f(t),t)^2
-2*N*V(f(t))
):

#S:= c*(1-2*h*R(t,r,th,ph))^(3/2)*(
#x*N*RicciScalar(g) 
#+x*N^(-1)*(EijEij-E^2) 
#+y*N^(-1)*diff(f(t),t)^2
#+z*2*N*V(f(t))
#):

#Ni_eq:= alpha(t,r,th,ph)=diff(R(t,r,th,ph),t)/H(t):
#N_eq:= first_order(a(t)^2*D2_(h*varpsi(t,r,th,ph)) +a(t)^2*D2_(h*R(t,r,th,ph))/H(t) - a(t)^2*diff(f(t),t)^2/2/H(t)^3*diff(h*R(t,r,th,ph),t)):

# Check first order action is zero

# Turn N_eq into one involving D2R and D2varpsi
N_eq;
elim_using_from(diff(R(t,r,th,ph),r$2),D2R=first_order(D2_(h*R(t,r,th,ph))),%);
elim_using_from(diff(varpsi(t,r,th,ph),r$2),D2varpsi=first_order(D2_(h*varpsi(t,r,th,ph))),%);
N_eq_ := expand(simplify(%));


# Evaluate the first order part of the action, and exchange spatial derivatives
first_order(S);
elim_using_from(diff(R(t,r,th,ph),r$2),D2R=first_order(D2_(h*R(t,r,th,ph))),%):
elim_using_from(diff(varpsi(t,r,th,ph),r$2),D2varpsi=first_order(D2_(h*varpsi(t,r,th,ph))),%):
expand(simplify(%));

elim_using_from(alpha(t,r,th,ph), Ni_eq, %):
elim_using_from(varpsi(t,r,th,ph), N_eq_, %):
subsH(expand(simplify(%))):
elim_using_from(V(f(t)),E00,%):
expand(simplify(%));

ans:=subs(diff(R(t,r,th,ph),t)=dR,R(t,r,th,ph)=R,%);

# Integrate the dR term by parts:
ans - dR*coeff(ans,dR) - R*diff(coeff(ans,dR),t);
subsH(%);
simplify(elim_using_from(diff(H(t),t),curv,%));
# From this, we can state that D2varpsi and D2R integrate spatially to give zero by the divergence theorem



# Now examine the second order
subsall(second_order(S)):
trig_simp(%):

[%,Ni_eq,N_eq,DRDR_eq]:



convert(%,D);
conv := subs([
alpha(t,r,th,ph) = alpha,
R(t,r,th,ph) = R,
varpsi(t,r,th,ph) = p,
D[4](R)(t,r,th,ph) = R4,
D[3](R)(t,r,th,ph) = R3,
D[2](R)(t,r,th,ph) = R2,
D[1](R)(t,r,th,ph) = Rt,
D[4,4](R)(t,r,th,ph) = R44,
D[3,3](R)(t,r,th,ph) = R33,
D[2,2](R)(t,r,th,ph) = R22,
D[3,4](R)(t,r,th,ph) = R34,
D[2,4](R)(t,r,th,ph) = R24,
D[2,3](R)(t,r,th,ph) = R23,
D[4](varpsi)(t,r,th,ph) = p4,
D[3](varpsi)(t,r,th,ph) = p3,
D[2](varpsi)(t,r,th,ph) = p2,
D[4,4](varpsi)(t,r,th,ph) = p44,
D[3,3](varpsi)(t,r,th,ph) = p33,
D[2,2](varpsi)(t,r,th,ph) = p22,
D[3,4](varpsi)(t,r,th,ph) = p34,
D[2,4](varpsi)(t,r,th,ph) = p24,
D[2,3](varpsi)(t,r,th,ph) = p23
],%):

conv[1];

# Remove alpha
elim_using_from(alpha,conv[2],%);
%;

# Remove all mixed derivative terms
int_by_parts(%,p22,p33,p3,p223,th): int_by_parts(%,p3,p223,p23,p23,r):
int_by_parts(%,p22,p44,p4,p224,ph): int_by_parts(%,p4,p224,p24,p24,r):
int_by_parts(%,p33,p44,p4,p334,ph): int_by_parts(%,p4,p334,p34,p34,th):

int_by_parts(%,p3,p23,p2,p33,th):
int_by_parts(%,p4,p34,p3,p44,ph):
int_by_parts(%,p4,p24,p2,p44,ph):

kinetic_trick(%,p2,p22,r):
kinetic_trick(%,p3,p33,th):
kinetic_trick(%,p4,p44,ph):

int_by_parts(%,p2,R2,R,p22,r):
int_by_parts(%,p3,R3,R,p33,th):
int_by_parts(%,p4,R4,R,p44,ph):

elim_using_from(p22,conv[3],%):

int_by_parts(%,p2,p33,p3,p23,th):
int_by_parts(%,p3,p44,p4,p34,ph):
int_by_parts(%,p2,p44,p4,p24,ph):

kinetic_trick(%,p3,p23,r):
kinetic_trick(%,p4,p24,r):
kinetic_trick(%,p3,p34,ph):
kinetic_trick(%,p4,p34,th):

int_by_parts_(%,p2,p,p22,r):
int_by_parts_(%,p3,p,p33,th):
int_by_parts_(%,p4,p,p44,ph):
elim_using_from(p22,conv[3],%):

convert(expand(simplify(%)),diff);



int_by_parts(%,R,R44,R4,R4,ph):
int_by_parts(%,R,R33,R3,R3,th):
int_by_parts(%,R,R22,R2,R2,r):

int_by_parts(%,Rt,R44,R4,R4t,ph):
kinetic_trick(%,R4,R4t,t):
int_by_parts(%,Rt,R33,R3,R3t,th):
kinetic_trick(%,R3,R3t,t):
int_by_parts(%,Rt,R22,R2,R2t,r):
kinetic_trick(%,R2,R2t,t):

int_by_parts(%,p,R44,R4,p4,ph):
int_by_parts(%,p4,R4,R,p44,ph):
int_by_parts(%,p,R33,R3,p3,th):
int_by_parts(%,p3,R3,R,p33,th):
int_by_parts(%,p,R22,R2,p2,r):
int_by_parts(%,p2,R2,R,p22,r):
elim_using_from(p22,conv[3],%):

kinetic_trick(%,R,Rt,t):
int_by_parts(%,R,R44,R4,R4,ph):
int_by_parts(%,R,R33,R3,R3,th):
int_by_parts(%,R,R22,R2,R2,r):



subsH(subsV(%));
convert(%,diff):
##elim_using_from(R22,conv[5],%):
elim_using_from(V(f(t)),E00,%):
elim_using_from(dVdphi(f(t)),cons0,%):
elim_using_from(diff(H(t),t),curv,%):
trig_simp(expand(simplify(%),symbolic)): 
#---------
%;
subs(p=deltap+R/H(t),%);
elim_using_from(R4^2,conv[4],%);
kinetic_trick(%,R,Rt,t);
subsall(%);
elim_using_from(diff(H(t),t),curv,%);

% - coeff(%,Rt,2)*Rt^2 + coeff(%,Rt,2) * (deltaRt^2 - (Rt - R/H(t)*k/a(t)^2)^2 + Rt^2):
expand(%);
kinetic_trick(%,R,Rt,t);
subsall(%);
elim_using_from(diff(H(t),t),curv,%);
subs(Rt = deltaRt +  R/H(t)*k/a(t)^2, %);
expand(%);

# This gives you equation (25)
expand(%/c);
