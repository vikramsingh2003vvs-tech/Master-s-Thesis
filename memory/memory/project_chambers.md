---
name: Chambers fit project
description: File chiave e architettura del fit Chambers numerico su dati Ataei per estrarre Gamma_is e Gamma_an
type: project
originSessionId: aaa2c983-6b61-4928-8f6c-5a3753c08b2d
---
**File principali** (tutti in `jupyter tesi/`):
- `Fit_Ataei_PyTorch.ipynb` — nuovo notebook (2026-04-28): stessa struttura di `Fit_Ataei.ipynb` ma con multistart PyTorch per i fit globali. Gradino: grid sweep GPU (30k starts) + L-BFGS-B con autograd. Bessel: joblib parallelo su CPU (n_jobs=-1). `chambers_lib.MR_Bessel` non portabile su GPU (usa scipy.special.iv), rimane su CPU.
- `Fit_Ataei.ipynb` — fit globali con scipy least_squares (riferimento estetico e per fit indipendenti).
- `chambers_tb.py` — integrale di Chambers numerico su FS tight-binding (Mirarchi A153). Funzione chiave: `MR_chambers_tb(B_arr, G_is, G_an, ...)`. N_orbit_max=12 default è sufficiente perché ωcτ<<1 per i parametri cuprati.
- `chambers_lib.py` — modello Bessel analitico (`MR_Bessel`) + utility loader Ataei. Da NON usare per il fit finale (è l'approssimazione ciclotrone).
- `Estrazione_Gamma_Ataei_TB.ipynb` — notebook creato in sessione, fa il fit con `ChambersFitEngine` (Chambers numerico, FS TB, 2 parametri liberi: Gamma_is, Gamma_an).
- `Estrazione_Gamma_Ataei.ipynb` — vecchio notebook che usa MR_Bessel (SBAGLIATO per il task attuale).

**Dati**: `Data/Ataei/` — MOESM2 (Nd-LSCO T=4,40,100K), MOESM4 (LSCO S1), MOESM6 (T=30K entrambi). Colonne: B[T], rho(B)/rho(0). MR = rho/rho0 - 1.

**Parametri TB** (Nd-LSCO p=0.24, Mirarchi A153): t=435, tp=-50, tpp=38 meV. Ancora a T=30K: Gamma_is=17.7 meV, Gamma_an=39.0 meV.

**ChambersFitEngine** — ottimizzazione chiave: precomputa cs_1_is, cs_1_an (N×N) in __init__ in modo che cs_1 = G_is*cs_1_is + G_an*cs_1_an sia un'operazione O(N²) per chiamata. compute_MR è vettorizzato su B tramite exp(-cs_1/B) (array N_B×N×N). Multi-orbita con serie geometrica 1/(1-exp(-gamma_total/B)).

**Why:** L'utente vuole fit con la VERA FS (no approssimazione Bessel/ciclotrone), solo 2 parametri liberi.
