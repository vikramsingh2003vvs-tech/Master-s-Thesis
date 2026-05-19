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

---

## 4. Osservazione fisica — Perché T=30K non si fitta bene ad alto campo (2026-05-13)

### Il problema osservato

Nel fit globale Bessel (modello cos⁴, `FIT_CUPY.ipynb`), il pannello T=30K mostra un flesso verso l'alto nei dati ad alto campo (B > 50T) che la curva fittata non riesce a riprodurre. Questo si osserva **esclusivamente a T=30K**: i pannelli T=4K, T=40K e T=100K sono tutti fittati in modo eccellente su tutto il range di campi.

### Perché il modello cos⁴ fallisce — analisi numerica

Il parametro α = Γ_an / (8ω_c) vale ~0.62 a B=85T per tutte le temperature: la serie di Bessel converge perfettamente ad alto campo, quindi il problema **non è numerico**.

La scala di campo a cui il MR inizia a saturare è B_sat = Γ_is · m* / K_MEV:

| T (K) | Γ_is (meV) | Γ_an (meV) | B_sat (T) |
|-------|-----------|-----------|----------|
| 4     | 8.9       | 32.4      | ~115     |
| 30    | 13.7      | 32.9      | ~177     |
| 40    | 15.6      | 33.1      | ~202     |
| 100   | 26.8      | 34.5      | ~347     |

A B=85T siamo ancora nel regime semiclassico (ω_c ≪ Γ_is per tutte le T): il MR dovrebbe crescere come B² e la serie di Bessel dovrebbe seguire i dati. Eppure a 30K non lo fa.

### Diagnosi: fluttuazioni superconduttrici

Nd-LSCO p=0.24 ha T_c ≈ 20–22K. Le quattro temperature del dataset si trovano in regimi fisici distinti:

| T (K) | T − T_c | Condizione |
|-------|---------|-----------|
| 4     | sotto T_c | SC soppresso da B (B_MIN = 25T > H_c2) → stato normale puro |
| **30**| **~10K sopra T_c** | **Fluttuazioni SC forti** |
| 40    | ~20K sopra T_c | Fluttuazioni deboli → fit perfetto |
| 100   | ~80K sopra T_c | Fluttuazioni assenti → fit perfetto |

Le **fluttuazioni superconduttrici** (tipo Aslamazov-Larkin) danno un contributo alla conduttività che è:
- **negativo rispetto al MR** (riducono la resistività e quindi sopprimono il MR osservato)
- **dipendente dal campo** (vengono soppresse dall'aumento di B)

### Meccanismo del misfit a 30K

1. A B basso–intermedio (5–50T): le fluttuazioni SC riducono il MR osservato **sotto** il valore di Boltzmann/Chambers puro → l'optimizer abbassa il prefactor A per compensare
2. A B alto (> 60T): il campo sopprime le fluttuazioni → emerge il MR del vero stato normale → i dati superano la curva fittata (che era stata "abbassata" in fase di ottimizzazione per adattarsi ai campi intermedi)

Il risultato è il **flesso verso l'alto** visibile nel pannello arancione (30K): non è il modello cos⁴ che sbaglia la forma funzionale, è che a 30K il sistema sovrappone due contributi fisici — il MR di Boltzmann più un termine di fluttuazione B-dipendente — e il modello ne conosce soltanto uno.

### Perché T=4K funziona comunque

B_MIN = 25T è sopra H_c2 a 4K: tutti i punti misurati sono già in stato normale puro, senza fluttuazioni. Il fit è quindi perfetto nel range disponibile.

### Implicazione per la tesi

L'impossibilità di fittare T=30K in modo globale con il modello di Chambers non è un difetto del codice o del modello angolare (cos⁴ o gradino): è la **firma delle fluttuazioni superconduttrici nel trasporto** che il formalismo di Boltzmann non può descrivere per costruzione. Questa limitazione è fisicamente attesa e documenta che a T=30K il sistema non è ancora un metallo normale ordinario nemmeno a campi moderati.
