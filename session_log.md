# Session Log — FIT_CUPY & chambers_lib

Data: 2026-05-11

---

## 1. Aggiunta celle markdown con le formule di fit (FIT_CUPY.ipynb)

Inserita una cella markdown subito dopo il titolo `# CUDA FIT BESSEL` con la documentazione completa dei due modelli utilizzati.

### Modello cos⁴ / Bessel
Il tasso di scattering ha simmetria tetragonale:

$$\Gamma(\varphi) = \Gamma_{is} + \Gamma_{an}\cos^4(2\varphi)$$

Decomposizione di Fourier:

$$\cos^4(2\varphi) = \frac{3}{8} + \frac{1}{2}\cos(4\varphi) + \frac{1}{8}\cos(8\varphi)$$

Scattering effettivo (media sull'orbita):

$$\tilde{\Gamma} = \Gamma_{is} + \frac{3}{8}\,\Gamma_{an}$$

Formula di Chambers (convenzione hole-like, barra = coniugato complesso):

$$\sigma_{xx} + i\,\sigma_{xy} = \pi\,\overline{\sum_t (-1)^t \frac{(\tilde{\Gamma} - i\,\omega_c(4t+1))\,\Omega(t)}{\tilde{\Gamma}^2 + \omega_c^2(4t+1)^2}}$$

con $\omega_c = K \cdot B / m^*$, $K = 0.116$ meV/T, e $\Omega(t)$ combinazione di $I_n(\alpha)$ e $I_n(\alpha/8)$ con parametro di Bessel $\alpha = \Gamma_{an}/(8\omega_c)$.

**Nota divergenza Bessel:** quando $B \to 0$, $\alpha \to \infty$ e le $I_n(\alpha) \sim e^\alpha/\sqrt{2\pi\alpha}$ divergono. Si impone un campo minimo:

$$B_\text{ref} = \frac{\Gamma_{an}\,m^*}{8\,K\,\alpha_\text{ref}}, \qquad \alpha_\text{ref} = 10$$

Sotto $B_\text{ref}$, $\sigma_{xx}$ e MR non vengono calcolati.

### Modello a Gradino
La superficie di Fermi (quadrante $[0, \pi/4]$) è divisa in:
- **Cold spot** (ampiezza $\varphi^*$): $\Gamma_1$
- **Hot spot** (ampiezza $\psi^* = \pi/4 - \varphi^*$): $\Gamma_2$

**Tempi di percorrenza ciclotronici** (con $S_i$ associato a $\Gamma_i$):

$$S_1 = \frac{2\varphi^*}{\omega}, \qquad S_2 = \frac{2\psi^*}{\omega}$$

Formula:

$$\sigma_{xx} = \mathrm{Re}\!\left[\overline{O_{I1} + O_{I2}}\right]$$

$$O_{I1} = \frac{4}{\Gamma_1^2 + \omega^2}(\Gamma_1 - i\omega)(\varphi^* + Z_{12}\,\omega\,t_1)$$

$$O_{I2} = \frac{4}{\Gamma_2^2 + \omega^2}(\Gamma_2 - i\omega)(\psi^* + Z_{21}\,\omega\,t_2)$$

$$D = 1 + i\,e^{-(\Gamma_1 S_1 + \Gamma_2 S_2)}$$

$$Z_{12} = \frac{1 - e^{-(\Gamma_2+i\omega)S_2}}{D}\!\left(\frac{1}{\Gamma_2+i\omega} - \frac{1}{\Gamma_1+i\omega}\right), \qquad t_1 = \frac{1 - e^{-(\Gamma_1+i\omega)S_1}}{2}$$

$$Z_{21} = \frac{1 - e^{-(\Gamma_1+i\omega)S_1}}{D}\!\left(\frac{1}{\Gamma_1+i\omega} - \frac{1}{\Gamma_2+i\omega}\right), \qquad t_2 = \frac{1 - e^{-(\Gamma_2+i\omega)S_2}}{2}$$

Limite $B \to 0$: $\sigma_{xx}(0) = 4\varphi^*/\Gamma_1 + 4\psi^*/\Gamma_2$, approssimato numericamente a $B = 10^{-8}$ T.

---

## 2. Correzione convenzione S₁/S₂ nel modello a Gradino

**Problema:** nel codice originale `S1 = 2ψ*/ω` e `S2 = 2φ*/ω`, cioè `S1` era associato al hot spot (Γ₂) e `S2` al cold spot (Γ₁) — convenzione opposta a quella attesa.

**Principio corretto:** `Sᵢ` deve essere il tempo di percorrenza ciclotronico della sezione con scattering `Γᵢ`.

**File modificati:**

### `chambers_lib.py` — `_sigma_complex_step`
```python
# Prima (sbagliato)
S1 = 2 * psi_star / w   # hot-spot
S2 = 2 * phi_star  / w  # cold-spot
denom = 1 + 1j * np.exp(-(Gamma1 * S2 + Gamma2 * S1))
Z12 = ((1 - np.exp(-(Gamma2 + 1j*w) * S1)) / denom * ...)
Z21 = ((1 - np.exp(-(Gamma1 + 1j*w) * S2)) / denom * ...)
t1  = (1 - np.exp(-(Gamma1 + 1j*w) * S2)) / 2.0
t2  = (1 - np.exp(-(Gamma2 + 1j*w) * S1)) / 2.0

# Dopo (corretto)
S1 = 2 * phi_star  / w   # cold-spot (Gamma1)
S2 = 2 * psi_star  / w   # hot-spot  (Gamma2)
denom = 1 + 1j * np.exp(-(Gamma1 * S1 + Gamma2 * S2))
Z12 = ((1 - np.exp(-(Gamma2 + 1j*w) * S2)) / denom * ...)
Z21 = ((1 - np.exp(-(Gamma1 + 1j*w) * S1)) / denom * ...)
t1  = (1 - np.exp(-(Gamma1 + 1j*w) * S1)) / 2.0
t2  = (1 - np.exp(-(Gamma2 + 1j*w) * S2)) / 2.0
```

### `FIT_CUPY.ipynb` — `cp_sigma_xx_step`
Stessa correzione applicata alla versione CUDA vettorializzata (`phi_gpu`↔`psi_gpu` per S1/S2, e tutti i riferimenti a S1/S2 nelle righe successive).

### Cella markdown del Gradino (cell `575c6d32`)
Aggiornata la definizione:
$$S_1 = \frac{2\varphi^*}{\omega}\ (\text{cold, }\Gamma_1), \qquad S_2 = \frac{2\psi^*}{\omega}\ (\text{hot, }\Gamma_2)$$
e tutte le formule di $D$, $Z_{12}$, $Z_{21}$, $t_1$, $t_2$.

**Nota:** la correzione è puramente convenzionale — il risultato numerico è identico perché i valori fisici (Γ₁ × tempo cold, Γ₂ × tempo hot) non cambiano.

---

## 3. Aggiunta selezione linear/quadratic nella sezione T-dipendenza (FIT_CUPY.ipynb)

La sezione "Fit con T-dependece esplicito" usa un modello di saturazione Mott-Ioffe-Regel:

$$\Gamma_\text{ideal}(T) = G_0 + \alpha \cdot T^p \quad (p=1\ \text{o}\ 2)$$

$$\Gamma(T) = \frac{\Gamma_\text{ideal}(T) \cdot G_{sat}}{\Gamma_\text{ideal}(T) + G_{sat}}$$

### Modifiche applicate

**Cella 15 (config)** — aggiunto all'inizio, sopra `bnds_thermo`:
```python
T_DEP = {
    "G_is": "linear",    # "linear" → G_0 + alpha·T  |  "quadratic" → G_0 + alpha·T²
    "G_an": "linear",
}
```

**Cella 16 (codice)** — 5 righe modificate:

1. Import aggiunto in cima:
```python
from chambers_lib import eval_T_dep
```

2. In `objective_thermo`:
```python
# Prima
G_ideal_is = G_is_0 + alpha_is * T
G_ideal_an = G_an_0 + alpha_an * T

# Dopo
G_ideal_is = eval_T_dep(T, alpha_is, G_is_0, T_DEP["G_is"])
G_ideal_an = eval_T_dep(T, alpha_an, G_an_0, T_DEP["G_an"])
```

3. Nel loop di estrazione `fit_global_bessel`:
```python
G_id_is = eval_T_dep(T, alpha_is_opt, G_is_0_opt, T_DEP["G_is"])
G_id_an = eval_T_dep(T, alpha_an_opt, G_an_0_opt, T_DEP["G_an"])
```

4. Nel plot della curva di saturazione (`T_smooth`):
```python
G_id_is_sm = eval_T_dep(T_smooth, alpha_is_opt, G_is_0_opt, T_DEP["G_is"])
G_id_an_sm = eval_T_dep(T_smooth, alpha_an_opt, G_an_0_opt, T_DEP["G_an"])
```

5. Nel display Markdown della formula estratta — reso dinamico:
```python
T_pow = {"linear": "T", "quadratic": "T^2"}
T_is_str = T_pow[T_DEP["G_is"]]
T_an_str = T_pow[T_DEP["G_an"]]
# La formula mostra {T_is_str} / {T_an_str} invece di T hardcoded
```

### Utilizzo
Per passare a quadratic basta modificare **una sola riga** in cella 15:
```python
T_DEP = {"G_is": "quadratic", "G_an": "quadratic"}
```
Tutto il resto (fit, estrazione, plot, formula Markdown) si adatta automaticamente.

> **Nota sui bounds:** con `"quadratic"`, il parametro `alpha` ha unità meV/K² invece di meV/K. Il bound `(0.01, 3.0)` potrebbe richiedere aggiustamento manuale in `bnds_thermo`.

---

## File modificati

| File | Modifiche |
|------|-----------|
| `FIT_CUPY.ipynb` | Celle markdown formule (nuove); fix S1/S2 in `cp_sigma_xx_step`; T_DEP config + 5 sostituzioni in cella T-dep |
| `chambers_lib.py` | Fix S1/S2 in `_sigma_complex_step` |
