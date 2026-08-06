(* ::Package:: *)

(* ========================================================================= *)
(* Friedmann Equation and ΛCDM Cosmology with Planck 2018 Parameters       *)
(* ========================================================================= *)

(* 
   This Mathematica package implements the Friedmann equations for the 
   ΛCDM (Lambda-Cold Dark Matter) cosmological model using the best-fit 
   parameters from Planck 2018 observations.
   
   Reference: Aghanim et al. 2020, "Planck 2018 results. VI. Cosmological parameters"
*)

BeginPackage["FriedmannLCDM`"]

(* ========================================================================= *)
(* Public Function Declarations *)
(* ========================================================================= *)

H0::usage = "Hubble constant in km s^-1 Mpc^-1"
h::usage = "Dimensionless Hubble parameter (H0/100)"
\[CapitalOmega]m::usage = "Matter density parameter today"
\[CapitalOmega]r::usage = "Radiation density parameter today"
\[CapitalOmega]\[CapitalLambda]::usage = "Dark energy density parameter today"
\[CapitalOmega]k::usage = "Curvature density parameter today"
c::usage = "Speed of light in km/s"

E::usage = "E[z] gives the dimensionless Hubble parameter as function of redshift"
HubbleRate::usage = "HubbleRate[z] gives H(z) in km s^-1 Mpc^-1"
DensityParameters::usage = "DensityParameters[z] gives {Ωr(z), Ωm(z), Ωk(z), ΩΛ(z)}"

ComovingDistanceIntegrand::usage = "ComovingDistanceIntegrand[z] gives 1/E(z)"
ComovingDistance::usage = "ComovingDistance[z] gives comoving distance χ(z)"
TransverseComovingDistance::usage = "TransverseComovingDistance[z] gives DM(z) in Mpc"
AngularDiameterDistance::usage = "AngularDiameterDistance[z] gives DA(z) in Mpc"
LuminosityDistance::usage = "LuminosityDistance[z] gives DL(z) in Mpc"
DistanceModulus::usage = "DistanceModulus[z] gives distance modulus μ(z)"
LookbackTime::usage = "LookbackTime[z] gives lookback time in Gyr"
ComovingVolumeElement::usage = "ComovingVolumeElement[z] gives dVc/(dz dΩ) in Mpc^3 sr^-1"

FriedmannFirst::usage = "FriedmannFirst[a] gives the first Friedmann equation H²(a)"
FriedmannSecond::usage = "FriedmannSecond[a] gives the second Friedmann equation (acceleration)"
ScaleFactor::usage = "ScaleFactor[z] converts redshift to scale factor: a = 1/(1+z)"
Redshift::usage = "Redshift[a] converts scale factor to redshift: z = 1/a - 1"

PlotHubbleRate::usage = "PlotHubbleRate[zmax] plots H(z) vs redshift"
PlotDensityEvolution::usage = "PlotDensityEvolution[zmax] plots density parameter evolution"
PlotDistances::usage = "PlotDistances[zmax] plots cosmological distances"

Begin["`Private`"]

(* ========================================================================= *)
(* Planck 2018 Best-Fit Parameters *)
(* ========================================================================= *)

(* Planck 2018 TT,TE,EE+lowE base-ΛCDM parameters *)
H0 = 67.36;  (* km s^-1 Mpc^-1 *)
h = 0.6736;  (* H0/100 *)

\[CapitalOmega]m = 0.3153;  (* Matter density parameter *)
\[CapitalOmega]k = 0.0;     (* Curvature parameter (flat universe) *)

(* Radiation density parameter calculation *)
\[CapitalOmega]\[Gamma]h2 = 2.4728*10^-5;  (* Photon density parameter * h^2 *)
Neff = 3.046;                              (* Effective number of neutrino species *)
\[CapitalOmega]r = \[CapitalOmega]\[Gamma]h2*(1 + 0.2271*Neff)/h^2 // N;  (* Total radiation *)

(* Dark energy density parameter (closure relation) *)
\[CapitalOmega]\[CapitalLambda] = 1 - \[CapitalOmega]m - \[CapitalOmega]k - \[CapitalOmega]r // N;

(* Speed of light *)
c = 299792.458;  (* km/s *)

(* Display parameters *)
Print["ΛCDM Cosmological Parameters (Planck 2018):"];
Print["H0 = ", H0, " km s^-1 Mpc^-1"];
Print["h = ", h];
Print["Ωm = ", \[CapitalOmega]m];
Print["Ωr = ", \[CapitalOmega]r];
Print["ΩΛ = ", \[CapitalOmega]\[CapitalLambda]];
Print["Ωk = ", \[CapitalOmega]k];
Print["Sum = ", \[CapitalOmega]m + \[CapitalOmega]r + \[CapitalOmega]\[CapitalLambda] + \[CapitalOmega]k];

(* ========================================================================= *)
(* Core Friedmann Equation Functions *)
(* ========================================================================= *)

(* Scale factor and redshift conversions *)
ScaleFactor[z_] := 1/(1 + z)
Redshift[a_] := 1/a - 1

(* Dimensionless Hubble parameter E(z) = H(z)/H0 *)
E[z_] := Sqrt[\[CapitalOmega]r*(1+z)^4 + \[CapitalOmega]m*(1+z)^3 + \[CapitalOmega]k*(1+z)^2 + \[CapitalOmega]\[CapitalLambda]]

(* Hubble rate H(z) *)
HubbleRate[z_] := H0*E[z]

(* First Friedmann equation: H²(a) *)
FriedmannFirst[a_] := H0^2*(\[CapitalOmega]r*a^-4 + \[CapitalOmega]m*a^-3 + \[CapitalOmega]k*a^-2 + \[CapitalOmega]\[CapitalLambda])

(* Second Friedmann equation: acceleration ä/a *)
FriedmannSecond[a_] := -(H0^2/2)*(2*\[CapitalOmega]r*a^-4 + \[CapitalOmega]m*a^-3 - 2*\[CapitalOmega]\[CapitalLambda])

(* ========================================================================= *)
(* Density Parameter Evolution *)
(* ========================================================================= *)

(* Individual density parameters as functions of redshift *)
\[CapitalOmega]rz[z_] := \[CapitalOmega]r*(1+z)^4/E[z]^2
\[CapitalOmega]mz[z_] := \[CapitalOmega]m*(1+z)^3/E[z]^2
\[CapitalOmega]kz[z_] := \[CapitalOmega]k*(1+z)^2/E[z]^2
\[CapitalOmega]\[CapitalLambda]z[z_] := \[CapitalOmega]\[CapitalLambda]/E[z]^2

(* Return all density parameters *)
DensityParameters[z_] := {\[CapitalOmega]rz[z], \[CapitalOmega]mz[z], \[CapitalOmega]kz[z], \[CapitalOmega]\[CapitalLambda]z[z]}

(* ========================================================================= *)
(* Cosmological Distances *)
(* ========================================================================= *)

(* Comoving distance integrand *)
ComovingDistanceIntegrand[z_] := 1/E[z]

(* Dimensionless comoving distance χ(z) *)
ComovingDistance[z_] := NIntegrate[1/E[zz], {zz, 0, z}, 
    Method -> "GlobalAdaptive", 
    WorkingPrecision -> 30, 
    AccuracyGoal -> 12, 
    PrecisionGoal -> 12]

(* Transverse comoving distance DM(z) in Mpc *)
TransverseComovingDistance[z_] := Which[
    \[CapitalOmega]k > 10^-10,  (* Open universe *)
    (c/H0)/Sqrt[\[CapitalOmega]k]*Sinh[Sqrt[\[CapitalOmega]k]*ComovingDistance[z]],
    
    \[CapitalOmega]k < -10^-10, (* Closed universe *)
    (c/H0)/Sqrt[-\[CapitalOmega]k]*Sin[Sqrt[-\[CapitalOmega]k]*ComovingDistance[z]],
    
    True,                       (* Flat universe *)
    (c/H0)*ComovingDistance[z]
]

(* Angular diameter distance DA(z) in Mpc *)
AngularDiameterDistance[z_] := TransverseComovingDistance[z]/(1+z)

(* Luminosity distance DL(z) in Mpc *)
LuminosityDistance[z_] := (1+z)*TransverseComovingDistance[z]

(* Distance modulus μ(z) *)
DistanceModulus[z_] := 5*Log10[LuminosityDistance[z]] + 25

(* Comoving volume element dVc/(dz dΩ) in Mpc^3 sr^-1 *)
ComovingVolumeElement[z_] := (c/H0)*TransverseComovingDistance[z]^2/E[z]

(* ========================================================================= *)
(* Time Relations *)
(* ========================================================================= *)

(* Lookback time in Gyr *)
(* Conversion factor: 1/H0 [km s^-1 Mpc^-1]^-1 = 9.778/h Gyr *)
LookbackTime[z_] := (9.778/h)*NIntegrate[1/((1+zz)*E[zz]), {zz, 0, z},
    Method -> "GlobalAdaptive",
    WorkingPrecision -> 20,
    AccuracyGoal -> 10,
    PrecisionGoal -> 10]

(* ========================================================================= *)
(* Plotting Functions *)
(* ========================================================================= *)

(* Plot Hubble rate evolution *)
PlotHubbleRate[zmax_: 5] := Plot[HubbleRate[z], {z, 0, zmax},
    PlotLabel -> "Hubble Rate H(z)",
    AxesLabel -> {"Redshift z", "H(z) [km s\!\(\*SuperscriptBox[\()\), \(-1\)]\) Mpc\!\(\*SuperscriptBox[\()\), \(-1\)]\)"],
    PlotStyle -> Thick,
    Frame -> True,
    GridLines -> Automatic]

(* Plot density parameter evolution *)
PlotDensityEvolution[zmax_: 10] := Plot[
    {\[CapitalOmega]rz[z], \[CapitalOmega]mz[z], \[CapitalOmega]\[CapitalLambda]z[z]}, {z, 0, zmax},
    PlotLabel -> "Density Parameter Evolution",
    AxesLabel -> {"Redshift z", "Ω(z)"},
    PlotLegends -> {"Ωr(z)", "Ωm(z)", "ΩΛ(z)"},
    PlotStyle -> {Red, Blue, Green},
    Frame -> True,
    GridLines -> Automatic,
    PlotRange -> {0, 1}]

(* Plot cosmological distances *)
PlotDistances[zmax_: 5] := Plot[
    {AngularDiameterDistance[z], LuminosityDistance[z], (c/H0)*ComovingDistance[z]}, 
    {z, 0, zmax},
    PlotLabel -> "Cosmological Distances",
    AxesLabel -> {"Redshift z", "Distance [Mpc]"},
    PlotLegends -> {"DA(z)", "DL(z)", "DC(z)"},
    PlotStyle -> {Blue, Red, Green},
    Frame -> True,
    GridLines -> Automatic,
    PlotRange -> All]

(* ========================================================================= *)
(* Analytical Solutions and Special Cases *)
(* ========================================================================= *)

(* Age of the universe in Gyr *)
AgeOfUniverse[] := LookbackTime[Infinity]

(* Hubble time in Gyr *)
HubbleTime[] := 9.778/h

(* Critical density in units of 10^-29 g cm^-3 *)
CriticalDensity[] := 3*H0^2/(8*Pi*6.67430*10^-11*3.08567758149137*10^22)

(* Matter-radiation equality redshift *)
MatterRadiationEquality[] := \[CapitalOmega]m/\[CapitalOmega]r - 1

(* Matter-dark energy equality redshift *)
MatterDarkEnergyEquality[] := (\[CapitalOmega]\[CapitalLambda]/\[CapitalOmega]m)^(1/3) - 1

Print["\nKey Cosmological Epochs:"];
Print["Matter-Radiation Equality: z_eq = ", MatterRadiationEquality[] // N];
Print["Matter-Dark Energy Equality: z_ΛM = ", MatterDarkEnergyEquality[] // N];
Print["Current age of universe: ", HubbleTime[], " Gyr"];

End[]
EndPackage[]

(* ========================================================================= *)
(* Usage Examples *)
(* ========================================================================= *)

Print["\n========================================"];
Print["Usage Examples:"];
Print["========================================"];

Print["Hubble rate at z=1: ", HubbleRate[1], " km s^-1 Mpc^-1"];
Print["Angular diameter distance to z=1: ", AngularDiameterDistance[1] // N, " Mpc"];
Print["Luminosity distance to z=1: ", LuminosityDistance[1] // N, " Mpc"];
Print["Distance modulus at z=1: ", DistanceModulus[1] // N];
Print["Lookback time to z=1: ", LookbackTime[1] // N, " Gyr"];

Print["\nDensity parameters at z=1:"];
{Ωr1, Ωm1, Ωk1, ΩΛ1} = DensityParameters[1];
Print["Ωr(z=1) = ", Ωr1 // N];
Print["Ωm(z=1) = ", Ωm1 // N];  
Print["ΩΛ(z=1) = ", ΩΛ1 // N];
Print["Sum = ", (Ωr1 + Ωm1 + Ωk1 + ΩΛ1) // N];

Print["\nTo create plots, use:"];
Print["PlotHubbleRate[5]"];
Print["PlotDensityEvolution[10]"];  
Print["PlotDistances[5]"];