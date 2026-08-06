I'm analysing Planck data, but am confused by the naming of the nuisance parameters

One code has these names:
```
'ycal':nuisance_params[0],
'A_cib_217':nuisance_params[1],
'xi_sz_cib':nuisance_params[2],
'A_sz':nuisance_params[3],
'ps_A_100_100':nuisance_params[4],
'ps_A_143_143':nuisance_params[5],
'ps_A_143_217':nuisance_params[6],
'ps_A_217_217':nuisance_params[7],
'ksz_norm':nuisance_params[8],
'gal545_A_100':nuisance_params[9],
'gal545_A_143':nuisance_params[10],
'gal545_A_143_217':nuisance_params[11],
'gal545_A_217':nuisance_params[12],
'galf_TE_A_100':nuisance_params[13],
'galf_TE_A_100_143':nuisance_params[14],
'galf_TE_A_100_217':nuisance_params[15],
'galf_TE_A_143':nuisance_params[16],
'galf_TE_A_143_217':nuisance_params[17],
'galf_TE_A_217':nuisance_params[18],
'calib_100T':nuisance_params[19],
'calib_217T':nuisance_params[20],
'cib_index':-1.3, #no range given in table so assume fixed
#-------------------------------------------------------------------
# These are all set to 1, so assume that these are fixed -----------
'A_cnoise_e2e_100_100_EE':1.,
'A_cnoise_e2e_143_143_EE':1.,
'A_cnoise_e2e_217_217_EE':1.,
'A_sbpx_100_100_TT':1.,
'A_sbpx_143_143_TT':1.,
'A_sbpx_143_217_TT':1.,
'A_sbpx_217_217_TT':1.,
'A_sbpx_100_100_EE':1.,
'A_sbpx_100_143_EE':1.,
'A_sbpx_100_217_EE':1.,
'A_sbpx_143_143_EE':1.,
'A_sbpx_143_217_EE':1.,
'A_sbpx_217_217_EE':1.,
'A_pol':1,
'A_planck':1.,
#-------------------------------------------------------------------
# These are fixed from Planck 2018 Likelihood Paper, Table 16 ------
'galf_EE_A_100':0.055,
'galf_EE_A_100_143':0.040,
'galf_EE_A_100_217':0.094,
'galf_EE_A_143':0.086,
'galf_EE_A_143_217':0.21,
'galf_EE_A_217':0.70,
'calib_100P':1.021,
'calib_143P':0.966,
'calib_217P':1.04,
#-------------------------------------------------------------------
# These are fixed from Planck 2018 Likelihood Paper, pg 39 ---------
'galf_EE_index':-2.4,
'galf_TE_index':-2.4,
```

But the original Planck papers have these names:
```
    1  0.2263225E-01   omegabh2              \Omega_b h^2
    2  0.1179211E+00   omegach2              \Omega_c h^2
    3  0.1041187E+01   theta                 100\theta_{MC}
    4  0.4949074E-01   tau                   \tau
    5 -0.4376602E-01   omegak                \Omega_K
   17  0.3030413E+01   logA                  {\rm{ln}}(10^{10} A_s)
   18  0.9723526E+00   ns                    n_s
   25  0.1000117E+01   calPlanck             y_{\rm cal}

   26  0.4211941E+02   acib217               A^{CIB}_{217}
   28  0.9998674E+00   xi                    \xi^{tSZ-CIB}
   29  0.6823064E+01   asz143                A^{tSZ}_{143}
   30  0.2377409E+03   aps100                A^{PS}_{100}
   31  0.4845758E+02   aps143                A^{PS}_{143}
   32  0.5638717E+02   aps143217             A^{PS}_{143\times217}
   33  0.1235650E+03   aps217                A^{PS}_{217}
   34  0.7909734E-04   aksz                  A^{kSZ}
   35  0.8766278E+01   kgal100               A^{{\rm dust}TT}_{100}
   36  0.1068038E+02   kgal143               A^{{\rm dust}TT}_{143}
   37  0.1969199E+02   kgal143217            A^{{\rm dust}TT}_{143\times217}
   38  0.9560933E+02   kgal217               A^{{\rm dust}TT}_{217}
   46  0.1138398E+00   galfTE100             A^{{\rm dust}TE}_{100}
   47  0.1348580E+00   galfTE100143          A^{{\rm dust}TE}_{100\times143}
   48  0.4809662E+00   galfTE100218          A^{{\rm dust}TE}_{100\times217}
   49  0.2234534E+00   galfTE143             A^{{\rm dust}TE}_{143}
   50  0.6611594E+00   galfTE143217          A^{{\rm dust}TE}_{143\times217}
   51  0.2054784E+01   galfTE217             A^{{\rm dust}TE}_{217}
   66  0.9997697E+00   cal0                  c_{100}
   67  0.9980799E+00   cal2                  c_{217}

    6  0.6000000E-01   mnu                   \Sigma m_\nu
    7  0.0000000E+00   meffsterile           m_{\nu,{\rm{sterile}}}^{\rm{eff}}
    8 -0.1000000E+01   w                     w
    9  0.0000000E+00   wa                    w_a
   10  0.3046000E+01   nnu                   N_{eff}
   11  0.2400000E+00   yhe                   Y_{He}
   12  0.0000000E+00   alpha1                \alpha_{-1}
   13  0.5000000E+00   deltazrei             {\Delta}z_{\rm re}
   14  0.1000000E+01   Alens                 A_{L}
   15 -0.1000000E+01   Alensf                A^f_L
   16  0.0000000E+00   fdm                   \epsilon_0 f_d
   19  0.0000000E+00   nrun                  n_{\rm run}
   20  0.0000000E+00   nrunrun               n_{\rm run, run}
   21  0.0000000E+00   r                     r
   22  0.0000000E+00   nt                    n_t
   23  0.0000000E+00   ntrun                 n_{t,{\rm run}}
   24  0.1000000E+01   Aphiphi               A^{\phi\phi}_L
   27 -0.1300000E+01   ncib                  n^{CIB}
   39  0.5500000E-01   galfEE100             A^{{\rm dust}EE}_{100}
   40  0.4000000E-01   galfEE100143          A^{{\rm dust}EE}_{100\times143}
   41  0.9400000E-01   galfEE100217          A^{{\rm dust}EE}_{100\times217}
   42  0.8600000E-01   galfEE143             A^{{\rm dust}EE}_{143}
   43  0.2100000E+00   galfEE143217          A^{{\rm dust}EE}_{143\times217}
   44  0.7000000E+00   galfEE217             A^{{\rm dust}EE}_{217}
   45 -0.2400000E+01   galfEEindex           index^{{\rm dust}EE}
   52 -0.2400000E+01   galfTEindex           index^{{\rm dust}TE}
   53  0.1000000E+01   Acnoisee2e100         A^{{\rm cnoise}EE}_{100}
   54  0.1000000E+01   Acnoisee2e143         A^{{\rm cnoise}EE}_{143}
   55  0.1000000E+01   Acnoisee2e217         A^{{\rm cnoise}EE}_{217}
   56  0.1000000E+01   Asbpx100TT            A^{{\rm subpix}TT}_{100}
   57  0.1000000E+01   Asbpx143TT            A^{{\rm subpix}TT}_{143}
   58  0.1000000E+01   Asbpx143217TT         A^{{\rm subpix}TT}_{143\times217}
   59  0.1000000E+01   Asbpx217TT            A^{{\rm subpix}TT}_{217}
   60  0.1000000E+01   Asbpx100EE            A^{{\rm subpix}EE}_{100}
   61  0.1000000E+01   Asbpx100143EE         A^{{\rm subpix}EE}_{100\times143}
   62  0.1000000E+01   Asbpx100217EE         A^{{\rm subpix}EE}_{100\times217}
   63  0.1000000E+01   Asbpx143EE            A^{{\rm subpix}EE}_{143}
   64  0.1000000E+01   Asbpx143217EE         A^{{\rm subpix}EE}_{143\times217}
   65  0.1000000E+01   Asbpx217EE            A^{{\rm subpix}EE}_{217}
   68  0.1021000E+01   calEE0                c_{EE100}
   69  0.9660000E+00   calEE1                c_{EE143}
   70  0.1040000E+01   calEE2                c_{EE217}
   71  0.1000000E+01   calPol                y_{\rm calPol}
   72  0.5408813E+02   H0                    H_0
   73  0.5611229E+00   omegal                \Omega_\Lambda
   74  0.4826431E+00   omegam                \Omega_m
   75  0.1411985E+00   omegamh2              \Omega_m h^2
   76  0.6451439E-03   omeganuh2             \Omega_\nu h^2
   77  0.7637164E-01   omegamh3              \Omega_m h^3
   78  0.7749885E+00   sigma8                \sigma_8
   79  0.9829867E+00   S8                    S_8
   80  0.5384040E+00   s8omegamp5            \sigma_8 \Omega_m^{0.5}
   81  0.6459543E+00   s8omegamp25           \sigma_8 \Omega_m^{0.25}
   82  0.1053766E+01   s8h5                  \sigma_8/h^{0.5}
   83  0.7970039E+02   rdragh                r_{\rm drag} h
   84  0.2645648E+01   rmsdeflect            \langle d^2\rangle^{1/2}
   85  0.6962585E+01   zrei                  z_{\rm re}
   86  0.2070579E+01   A                     10^9 A_s
   87  0.1875446E+01   clamp                 10^9 A_s e^{-2\tau}
   88  0.1204906E+04   DL40                  D_{40}
   89  0.5747577E+04   DL220                 D_{220}
   90  0.2535425E+04   DL810                 D_{810}
   91  0.8169539E+03   DL1420                D_{1420}
   92  0.2333263E+03   DL2000                D_{2000}
   93  0.9723526E+00   ns02                  n_{s,0.002}
   94  0.2454914E+00   yheused               Y_P
   95  0.2468183E+00   YpBBN                 Y_P^{\rm{BBN}}
   96  0.2538746E+01   DHBBN                 10^5D/H
   97  0.1533136E+02   age                   {\rm{Age}}/{\rm{Gyr}}
   98  0.1089411E+04   zstar                 z_*
   99  0.1447711E+03   rstar                 r_*
  100  0.1041338E+01   thetastar             100\theta_*
  101  0.1390241E+02   DAstar                D_{\rm{M}}(z_*)/{\rm{Gpc}}
  102  0.1060390E+04   zdrag                 z_{\rm{drag}}
  103  0.1473528E+03   rdrag                 r_{\rm{drag}}
  104  0.1407863E+00   kd                    k_{\rm D}
  105  0.1605091E+00   thetad                100\theta_{\rm{D}}
  106  0.3358780E+04   zeq                   z_{\rm{eq}}
  107  0.1025131E-01   keq                   k_{\rm{eq}}
  108  0.8220640E+00   thetaeq               100\theta_{\rm{eq}}
  109  0.4537648E+00   thetarseq             100\theta_{\rm{s,eq}}
  110  0.6016921E+02   Hubble015             H(0.15)
  111  0.7890209E+03   DM015                 D_{\rm{M}}(0.15)
  112  0.7148324E+02   Hubble038             H(0.38)
  113  0.1840765E+04   DM038                 D_{\rm{M}}(0.38)
  114  0.7882205E+02   Hubble051             H(0.51)
  115  0.2358485E+04   DM051                 D_{\rm{M}}(0.51)
  116  0.8488069E+02   Hubble061             H(0.61)
  117  0.2723353E+04   DM061                 D_{\rm{M}}(0.61)
  118  0.2289221E+03   Hubble233             H(2.33)
  119  0.6356611E+04   DM233                 D_{\rm{M}}(2.33)
  120  0.5285279E+00   fsigma8z015           f\sigma_8(0.15)
  121  0.7017347E+00   sigma8z015            \sigma_8(0.15)
  122  0.5109941E+00   fsigma8z038           f\sigma_8(0.38)
  123  0.6069112E+00   sigma8z038            \sigma_8(0.38)
  124  0.4928247E+00   fsigma8z051           f\sigma_8(0.51)
  125  0.5618705E+00   sigma8z051            \sigma_8(0.51)
  126  0.4770650E+00   fsigma8z061           f\sigma_8(0.61)
  127  0.5308897E+00   sigma8z061            \sigma_8(0.61)
  128  0.2638434E+00   fsigma8z233           f\sigma_8(2.33)
  129  0.2639468E+00   sigma8z233            \sigma_8(2.33)
  130   2.511492E+01   f2000_143             f_{2000}^{143}
  131   2.916017E+01   f2000_x               f_{2000}^{143\times217}
  132   1.039389E+02   f2000_217             f_{2000}^{217}
  133  0.3955488E+03   chi2_simall           \chi^2_{\rm simall}
  134  0.2116458E+02   chi2_lowl             \chi^2_{\rm lowl}
  135  0.2336526E+04   chi2_plik             \chi^2_{\rm plik}
  136  0.1271102E+01   chi2_prior            \chi^2_{\rm prior}
  137  0.2753240E+04   chi2_CMB              \chi^2_{\rm CMB}

 -log(Like)     chi-sq   data
    197.774    395.549   CMB: simall = simall_100x143_offlike5_EE_Aplanck_B
     10.582     21.165   CMB: lowl = commander_dx12_v3_2_29
   1168.263   2336.526   CMB: plik = plik_rd12_HM_v22b_TTTEEE
                                                                                                                                                                                                   38,35          7%
                                                                                                                                                                                         ```
                                                                                                                                                                                        
Can you explain the difference?

-----------------------------------------------------------------

Okay, let's break down the differences between these two lists of Planck nuisance parameters.

**Short Answer:**

The two lists represent largely the **same underlying physical nuisance parameters** used in Planck analysis, but they differ in:

1.  **Naming Convention:** The first list (from your code) uses more descriptive, often longer names with underscores (e.g., `A_cib_217`). The second list (from Planck papers/chains) uses shorter, more compact names often used in the official likelihood code and MCMC outputs (e.g., `acib217`). The second list also provides the standard LaTeX mathematical notation (e.g., `A^{CIB}_{217}`).
2.  **Scope & Context:**
    *   The **first list** is a specific implementation *within a piece of code*. It defines which parameters are being treated as *variable* nuisance parameters (`nuisance_params[0]` to `nuisance_params[20]`) for *that specific analysis run*, and which are *fixed* either to 1.0, specific values from papers, or another constant (`cib_index`).
    *   The **second list** appears to be an excerpt from a `.paramnames` file or similar output associated with official Planck MCMC chains (like CosmoMC). It lists *all* parameters in that chain, including cosmological parameters (like `omegabh2`), nuisance parameters, derived parameters (like `H0`, `sigma8`), and even `chi-squared` values. The numerical values shown are likely the *best-fit* values from a particular analysis run for varied parameters, or the *fixed* values used for others.
3.  **Varied vs. Fixed Status:** The code explicitly shows how *it* handles parameters (varied array vs. fixed values). The Planck list shows the state *for a specific published analysis*, where some parameters were varied (and have best-fit values listed) and others were fixed (indicated by their fixed value being listed, often identical across different chains if fixed).

**Detailed Parameter Mapping and Explanation:**

Let's map the parameters from your code (List 1) to the Planck list (List 2) and explain them:

| Code Parameter (List 1)        | Planck Parameter (List 2) | LaTeX Symbol (List 2)        | Description & Notes                                                                                                                               | Status in Code                                | Status/Value in Planck List |
| :----------------------------- | :------------------------ | :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------- | :-------------------------- |
| `ycal`                         | `calPlanck` (25)          | `y_{\rm cal}`                | Overall calibration parameter, often related to Compton-Y parameter calibration for SZ.                                                           | Varied (`nuisance_params[0]`)                 | Varied (Best fit ~1.0001)   |
| `A_cib_217`                    | `acib217` (26)            | `A^{CIB}_{217}`              | Amplitude of the Cosmic Infrared Background power spectrum template at 217 GHz.                                                                   | Varied (`nuisance_params[1]`)                 | Varied (Best fit ~42.1)     |
| `xi_sz_cib`                    | `xi` (28)                 | `\xi^{tSZ-CIB}`              | Correlation coefficient between the thermal Sunyaev-Zel'dovich (tSZ) effect and CIB.                                                              | Varied (`nuisance_params[2]`)                 | Varied (Best fit ~0.9999)   |
| `A_sz`                         | `asz143` (29)             | `A^{tSZ}_{143}`              | Amplitude of the tSZ power spectrum template, conventionally defined at 143 GHz.                                                                  | Varied (`nuisance_params[3]`)                 | Varied (Best fit ~6.82)     |
| `ps_A_100_100`                 | `aps100` (30)             | `A^{PS}_{100}`               | Amplitude of the Poisson point source power spectrum at 100 GHz.                                                                                  | Varied (`nuisance_params[4]`)                 | Varied (Best fit ~237.7)    |
| `ps_A_143_143`                 | `aps143` (31)             | `A^{PS}_{143}`               | Amplitude of the Poisson point source power spectrum at 143 GHz.                                                                                  | Varied (`nuisance_params[5]`)                 | Varied (Best fit ~48.5)     |
| `ps_A_143_217`                 | `aps143217` (32)          | `A^{PS}_{143\times217}`      | Amplitude of the Poisson point source cross power spectrum between 143 and 217 GHz.                                                             | Varied (`nuisance_params[6]`)                 | Varied (Best fit ~56.4)     |
| `ps_A_217_217`                 | `aps217` (33)             | `A^{PS}_{217}`               | Amplitude of the Poisson point source power spectrum at 217 GHz.                                                                                  | Varied (`nuisance_params[7]`)                 | Varied (Best fit ~123.6)    |
| `ksz_norm`                     | `aksz` (34)               | `A^{kSZ}`                    | Amplitude of the kinetic Sunyaev-Zel'dovich (kSZ) power spectrum template.                                                                        | Varied (`nuisance_params[8]`)                 | Varied (Best fit ~0.000079) |
| `gal545_A_100`                 | `kgal100` (35)            | `A^{{\rm dust}TT}_{100}`     | Amplitude of Galactic thermal dust *Temperature* (TT) power spectrum at 100 GHz (often using 545/857 GHz maps as template).                           | Varied (`nuisance_params[9]`)                 | Varied (Best fit ~8.77)     |
| `gal545_A_143`                 | `kgal143` (36)            | `A^{{\rm dust}TT}_{143}`     | Amplitude of Galactic thermal dust TT power spectrum at 143 GHz.                                                                                  | Varied (`nuisance_params[10]`)                | Varied (Best fit ~10.68)    |
| `gal545_A_143_217`             | `kgal143217` (37)         | `A^{{\rm dust}TT}_{143\times217}` | Amplitude of Galactic thermal dust TT cross power spectrum between 143 and 217 GHz.                                                             | Varied (`nuisance_params[11]`)                | Varied (Best fit ~19.69)    |
| `gal545_A_217`                 | `kgal217` (38)            | `A^{{\rm dust}TT}_{217}`     | Amplitude of Galactic thermal dust TT power spectrum at 217 GHz.                                                                                  | Varied (`nuisance_params[12]`)                | Varied (Best fit ~95.61)    |
| `galf_TE_A_100`                | `galfTE100` (46)          | `A^{{\rm dust}TE}_{100}`     | Amplitude of Galactic dust *TE cross-correlation* power spectrum at 100 GHz.                                                                      | Varied (`nuisance_params[13]`)                | Varied (Best fit ~0.114)    |
| `galf_TE_A_100_143`            | `galfTE100143` (47)       | `A^{{\rm dust}TE}_{100\times143}` | Amplitude of Galactic dust TE cross-correlation power spectrum between 100 and 143 GHz.                                                         | Varied (`nuisance_params[14]`)                | Varied (Best fit ~0.135)    |
| `galf_TE_A_100_217`            | `galfTE100218` (48)       | `A^{{\rm dust}TE}_{100\times217}` | Amplitude of Galactic dust TE cross-correlation power spectrum between 100 and 217 GHz. (Note typo 218 vs 217)                                   | Varied (`nuisance_params[15]`)                | Varied (Best fit ~0.481)    |
| `galf_TE_A_143`                | `galfTE143` (49)          | `A^{{\rm dust}TE}_{143}`     | Amplitude of Galactic dust TE cross-correlation power spectrum at 143 GHz.                                                                      | Varied (`nuisance_params[16]`)                | Varied (Best fit ~0.223)    |
| `galf_TE_A_143_217`            | `galfTE143217` (50)       | `A^{{\rm dust}TE}_{143\times217}` | Amplitude of Galactic dust TE cross-correlation power spectrum between 143 and 217 GHz.                                                         | Varied (`nuisance_params[17]`)                | Varied (Best fit ~0.661)    |
| `galf_TE_A_217`                | `galfTE217` (51)          | `A^{{\rm dust}TE}_{217}`     | Amplitude of Galactic dust TE cross-correlation power spectrum at 217 GHz.                                                                      | Varied (`nuisance_params[18]`)                | Varied (Best fit ~2.055)    |
| `calib_100T`                   | `cal0` (66)               | `c_{100}`                    | Inter-frequency *Temperature* calibration factor for 100 GHz (relative to 143 GHz, often).                                                      | Varied (`nuisance_params[19]`)                | Varied (Best fit ~0.9998)   |
| `calib_217T`                   | `cal2` (67)               | `c_{217}`                    | Inter-frequency *Temperature* calibration factor for 217 GHz (relative to 143 GHz, often).                                                      | Varied (`nuisance_params[20]`)                | Varied (Best fit ~0.9981)   |
| `cib_index`                    | `ncib` (27)               | `n^{CIB}`                    | Effective spectral index for CIB frequency scaling.                                                                                               | Fixed (-1.3)                                  | Fixed (-1.3)                |
| `A_cnoise_e2e_XXX_XXX_EE`      | `Acnoisee2eXXX` (53-55)   | `A^{{\rm cnoise}EE}_{XXX}`   | Amplitude scaling for correlated noise templates in *EE polarization*.                                                                          | Fixed (1.0)                                   | Fixed (1.0)                 |
| `A_sbpx_..._TT` / `A_sbpx_..._EE` | `Asbpx...TT/EE` (56-65) | `A^{{\rm subpix}TT/EE}_{...}`| Amplitude scaling for sub-pixel leakage effects (from T to P, or beam mismatch) in TT or EE.                                                    | Fixed (1.0)                                   | Fixed (1.0)                 |
| `A_pol`                        | `calPol` (71)?            | `y_{\rm calPol}`?            | Unclear. `A_pol=1` might be a placeholder or unused. `calPol` is the overall polarization calibration. The code *does* use specific P calibs below. | Fixed (1.0)                                   | Varied (`calPol`, best fit ~1.0) or Fixed |
| `A_planck`                     | N/A (Likelihood related)  | N/A                          | Often a normalization factor for the overall likelihood, usually fixed to 1.0 when using actual Planck data.                                    | Fixed (1.0)                                   | N/A (Not a physical parameter) |
| `galf_EE_A_XXX` / `_XXX_YYY`   | `galfEE...` (39-44)       | `A^{{\rm dust}EE}_{...}`     | Amplitude of Galactic dust *EE auto/cross-correlation* power spectrum.                                                                          | Fixed (Paper values)                          | Fixed (Paper values)        |
| `calib_100P`                   | `calEE0` (68)             | `c_{EE100}`                  | Inter-frequency *Polarization* calibration factor for 100 GHz.                                                                                  | Fixed (1.021)                                 | Fixed (1.021)               |
| `calib_143P`                   | `calEE1` (69)             | `c_{EE143}`                  | Inter-frequency *Polarization* calibration factor for 143 GHz.                                                                                  | Fixed (0.966)                                 | Fixed (0.966)               |
| `calib_217P`                   | `calEE2` (70)             | `c_{EE217}`                  | Inter-frequency *Polarization* calibration factor for 217 GHz.                                                                                  | Fixed (1.04)                                  | Fixed (1.04)                |
| `galf_EE_index`                | `galfEEindex` (45)        | `index^{{\rm dust}EE}`       | Spectral index for Galactic dust EE foreground model.                                                                                             | Fixed (-2.4)                                  | Fixed (-2.4)                |
| `galf_TE_index`                | `galfTEindex` (52)        | `index^{{\rm dust}TE}`       | Spectral index for Galactic dust TE foreground model.                                                                                             | Fixed (-2.4)                                  | Fixed (-2.4)                |

**Summary of Differences:**

*   **Naming:** Code uses longer, descriptive names; Planck list uses short code names + LaTeX.
*   **Content:** Code list focuses *only* on parameters *it* uses/needs; Planck list is comprehensive for a specific analysis (cosmology, nuisance, derived, chi2).
*   **Status:** Code list *defines* which parameters are varied/fixed *for its run*; Planck list shows best-fit or fixed *values* from a *completed* analysis.

Essentially, your code snippet is setting up an analysis using a standard set of Planck nuisance parameters, deciding which ones to vary and which to fix based on common practices or specific goals documented in Planck papers (like the 2018 Likelihood Paper). The second list provides a reference point showing the names and typical values (either best-fit or fixed) for these parameters from an official Planck analysis.
