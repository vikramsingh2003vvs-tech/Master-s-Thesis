---
name: Ripresa sessione — Fit_Ataei_PyTorch
description: Stato al 2026-04-28 — nuovo notebook PyTorch multistart creato, pronto per essere eseguito sul nuovo PC con RTX 5070
type: project
originSessionId: dd6d154d-cb74-4a59-8644-a527c92f9de3
---
## File coinvolti
- `jupyter tesi/Fit_Ataei_PyTorch.ipynb` — **nuovo notebook** (55 celle), fit globali con multistart PyTorch
- `jupyter tesi/Fit_Ataei.ipynb` — riferimento originale (fit indipendenti + fit globali scipy, NON duplicare qui)
- `jupyter tesi/chambers_lib.py` — modelli `mr_step`, `MR_Bessel`, `build_global_residual`, `eval_T_dep`, `phi_tanh`, `log_bin`
- `jupyter tesi/vik_lib.py` — `font1`, `font2` per i plot

---

## Cosa fa il nuovo notebook

### Strategia multistart (perché)
`scipy.least_squares` con un singolo p0 si blocca in minimi locali (Gamma_an batte i bound a T=30K e T=40K). Con il nuovo PC (RTX 5070 + Intel Core Ultra 2 Gen 9) si vogliono provare migliaia di starting point in parallelo.

### Modello gradino — PyTorch GPU
- `_sigma_xx_step_batch(B, G1, G2, phi_star, m_star)` — porta `_sigma_complex_step` di `chambers_lib.py` su torch cdouble, broadcast `(N_starts, N_B)`. Differenziabile via autograd.
- `mr_step_batch(B, G1, G2, phi_star, m_star)` — MR = sig0/sigB - 1, batch.
- `build_loss_step_batch(...)` — factory loss MSE su dati log-binned, **8 params** `[a1,c1,a2,c2,phi0,dphi,T_phi,m_star]`
- `build_loss_step_batch_mfixed(...)` — stessa ma **7 params** (m* fissato)
- `run_multistart_torch(loss_fn, lb, ub, N_grid=30_000, N_refine=30)`:
  - **Fase 1**: 30k starts casuali → forward pass GPU in un colpo → prende top-30 per loss
  - **Fase 2**: scipy L-BFGS-B su ciascuno dei top-30 con **gradiente esatto da `loss.backward()`**

### Modello Bessel — joblib CPU
`MR_Bessel` usa `scipy.special.iv` (Bessel functions di ordine arbitrario) → non portabile su torch senza reimplementare Miller backward recurrence. Si usa joblib `n_jobs=-1` (tutti i core CPU) sia per la fase 1 (grid sweep) che per la fase 2 (raffinamento parallelo).
- `build_loss_bessel_np(DATA, B_MIN, T_LIST, unpack_fn)` — loss numpy
- `run_multistart_joblib(loss_fn, lb, ub, N_grid=3_000, N_refine=30, n_jobs=-1)`

---

## Sezioni del notebook

| Sezione | Campione | Modelli |
|---------|----------|---------|
| 2 | Nd-LSCO (MOESM2+6) | Gradino (PyTorch) + Bessel (joblib) |
| 3 | LSCO S1 (MOESM4+6) | Gradino (PyTorch) + Bessel (joblib) |
| 4 | Nd-LSCO calibrato A153 | Bessel 4 params: a_is, a_an, G_is(30), G_an(30) |
| 5 | Nd-LSCO m* fissata | Gradino 7 params (PyTorch) |

---

## Setup sul nuovo PC (prima esecuzione)

```bash
# 1. Installare PyTorch con CUDA 12.x per RTX 5070
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Verificare
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. joblib già presente in Anaconda
```

Sul PC attuale (vecchio) torch non funziona per conflitto `typing_extensions` — non preoccuparsi, il codice è corretto.

---

## Variabili di output standard (per plot/tabelle)

Dopo ogni fit globale le celle di plot si aspettano:

**Gradino (8 o 7 params)**
```python
a1_opt, c1_opt, a2_opt, c2_opt, phi0_opt, dphi_opt, T_phi_opt, m_star_opt
rms_step
```

**Bessel (5 params)**
```python
a_is_opt, c_is_opt, a_an_opt, c_an_opt, m_star_b_opt
rms_bessel
```

**Bessel calibrato A153 (4 params)**
```python
a_is_opt, a_an_opt, G_is_30_opt, G_an_30_opt
c_is_opt, c_an_opt  # calcolati da pin-to-30K
m_star_b_opt = 1.48784
rms_cal
```

---

## Prossimi step

1. **Eseguire sul nuovo PC** — primo run completo di tutte le sezioni
2. **Confrontare con Fit_Ataei.ipynb** — verificare se i fit globali migliorano (specialmente T=30K Nd-LSCO dove Gamma_an batteva il bound superiore)
3. **Se i risultati Bessel restano problematici**: considerare di implementare `iv(n, x)` via Miller backward recurrence in torch per avere anche il modello Bessel su GPU (non urgente)
4. **Eventuale tuning**: aumentare `N_grid` o `N_refine` se la qualità non soddisfa; ridurli se il tempo è troppo

---

## Note tecniche

- `log_bin(B, R, n_bins=60)` di `chambers_lib` viene usato per pre-campionare i dati: bilancia il peso tra bassi e alti B prima del fit
- `cl.PHI_MAX = np.pi/4 - 1e-4` — asintoto fisico per phi_star nel modello gradino
- Nel modello gradino: `phi_star` viene clampato a `[1e-6, PHI_MAX]` dentro la loss torch per evitare instabilità numeriche a phi=0
- `T_DEP["G2"] = "quadratic"` nella configurazione default (Sezione 2) — può cambiare nelle sezioni successive
