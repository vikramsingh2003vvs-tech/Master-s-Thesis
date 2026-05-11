"""
chambers_lib - modelli di Chambers per la magnetoresistenza orbitale.

Due modelli (parametri fisici):
  * gradino       : Gamma1, Gamma2 [meV],  phi_star [rad],  m_star [m_e]
  * cos^4/Bessel  : G_is,  G_an  [meV],                     m_star [m_e]

Convenzione:  A (prefattore globale di sigma_xx, sigma_xy) = 1.
  In  MR = sigma_0/sigma_xx - 1  A si cancella, per cui e' omesso.
  La convenzione hole-like e' ottenuta con  np.conj(.)  (flip di segno su sigma_xy).

Tutte le funzioni sono "pure": prendono i parametri espliciti, niente globali.

T-dipendenze (due API):
  * factory legacy  make_step_T_deps / make_bessel_T_deps: closures G(T), phi*(T)
    con parametri fissati da un singolo fit a T = T_exp.
  * helpers nuovi   eval_T_dep / fit_T_dep / phi_tanh / fit_phi_tanh:
    forma esplicita a 2-3 parametri (a, c)  oppure  (phi_0, dphi, T_phi),
    pensati per fit globali multi-T via build_global_residual.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import iv
import os
import pandas as pd
from numba import njit

# ---------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------
K_MEV = 0.116  # hbar*omega_c [meV] = K_MEV * B[T] / m_star[m_e] 
PHI_MAX = np.pi / 4 - 1e-4      # asintoto di phi*(T); evita la singolarita' a pi/4

# ---------------------------------------------------------------------
# Funzioni ausiliarie
# ---------------------------------------------------------------------
def com_omega(B, m_star):
    """Frequenza ciclotrone hbar*omega_c [meV]. B in T, m_star in m_e."""
    return K_MEV * B / m_star

def com_alpha(G_an, B, m_star):
    """Parametro di anisotropia Bessel:  alpha = G_an / (8 * hbar*omega_c)."""
    return G_an / (8.0 * com_omega(B, m_star))


# =====================================================================
# MODELLO 1 - GRADINO (Chambers step-function)
# =====================================================================
def _sigma_complex_step(B, Gamma1, Gamma2, phi_star, m_star):
    """sigma_xx + i*sigma_xy  (A=1 implicito).  Conv. hole: conj(.)."""
    psi_star = np.pi / 4 - phi_star
    w = com_omega(B, m_star)
    w = np.where(w == 0, 1e-12, w)
    S1 = 2 * psi_star / w
    S2 = 2 * phi_star  / w
    denom = 1 + 1j * np.exp(-(Gamma1 * S2 + Gamma2 * S1))
    Z12 = ((1 - np.exp(-(Gamma2 + 1j*w) * S1)) / denom
           * (1/(Gamma2 + 1j*w) - 1/(Gamma1 + 1j*w)))
    Z21 = ((1 - np.exp(-(Gamma1 + 1j*w) * S2)) / denom
           * (1/(Gamma1 + 1j*w) - 1/(Gamma2 + 1j*w)))
    t1  = (1 - np.exp(-(Gamma1 + 1j*w) * S2)) / 2.0
    t2  = (1 - np.exp(-(Gamma2 + 1j*w) * S1)) / 2.0
    O_I1 = (4 / (Gamma1**2 + w**2)) * (Gamma1 - 1j*w) * (phi_star + Z12 * w * t1)
    O_I2 = (4 / (Gamma2**2 + w**2)) * (Gamma2 - 1j*w) * (psi_star + Z21 * w * t2)
    return np.conj(O_I1 + O_I2)

def sigma_xx_step(B, Gamma1, Gamma2, phi_star, m_star):
    return np.real(_sigma_complex_step(B, Gamma1, Gamma2, phi_star, m_star))

def sigma_xy_step(B, Gamma1, Gamma2, phi_star, m_star):
    return np.imag(_sigma_complex_step(B, Gamma1, Gamma2, phi_star, m_star))

def mr_step(B, Gamma1, Gamma2, phi_star, m_star):
    """MR = sigma_0/sigma_xx - 1 (A si cancella)."""
    sig0 = sigma_xx_step(np.array([1e-8]), Gamma1, Gamma2, phi_star, m_star)[0]
    return sig0 / sigma_xx_step(B, Gamma1, Gamma2, phi_star, m_star) - 1.0


# =====================================================================
# MODELLO 2 - COS^4 / BESSEL
# =====================================================================
@njit(fastmath=True)
def _numba_t_loop(t_max, G_tilde, omega, mu_arr, N_all, W, I_al):
    alpha_len = len(omega)
    num_mu = len(mu_arr)
    num_N = len(N_all)
    
    # Array di output
    temp = np.zeros(alpha_len, dtype=np.complex128)
    
    for t in range(-t_max, t_max + 1):
        t_an = 4.0 * t + 1.0
        sign_t = 1.0 if t % 2 == 0 else -1.0
        
        for b in range(alpha_len):
            # Calcolo del denominatore per questo specifico campo B
            denom = G_tilde**2 + (omega[b] * t_an)**2
            
            # Somma interna su mu e N senza creare array 3D!
            sum_s_c_b = 0.0 + 0.0j
            for m in range(num_mu):
                mu_val = mu_arr[m]
                idx1 = abs(t - 2 * mu_val)
                iv1 = I_al[idx1, b]
                
                for n in range(num_N):
                    N_val = N_all[n]
                    idx2 = abs(t - 2 * mu_val + 2 * N_val)
                    iv2 = I_al[idx2, b]
                    
                    w_val = W[m, n, b]
                    sum_s_c_b += iv1 * iv2 * w_val
            
            # Aggiornamento dell'integrale
            inner_c = (G_tilde - 1j * omega[b] * t_an) * sum_s_c_b
            temp[b] += sign_t * inner_c / denom
            
    return temp

# =====================================================================
# 2. LA FUNZIONE WRAPPER PYTHON
# =====================================================================
def _sigma_Bessel_complex(B, G_is, G_an, m_star,
                          t_max=30, s_max=2, mu_max=10):
    """sigma_xx + i*sigma_xy nel modello cos^4/Bessel (Accelerazione NUMBA)."""
    B       = np.atleast_1d(B).astype(float)
    alpha   = com_alpha(G_an, B, m_star) # Assicurati che chiamino le tue def
    omega   = com_omega(B, m_star)
    G_tilde = G_is + 3.0 * G_an / 8.0
    
    # Preparazione array
    s_arr   = np.arange(-s_max, s_max + 1)
    sign_s  = (-1.0) ** s_arr
    N_all   = np.concatenate([2*s_arr, 2*s_arr + 1])
    mu_arr  = np.arange(-mu_max, mu_max + 1)
    sign_mu = (-1.0) ** mu_arr
    
    N_abs   = int(np.max(np.abs(N_all)))
    k_al    = np.arange(0, t_max + 2*mu_max + 2*N_abs + 1)
    k_al8   = np.arange(0, mu_max + N_abs + 1)
    
    # Chiamate a Bessel (SciPy C-level)
    I_al    = iv(k_al[:,  None], alpha[None, :])
    I_al8   = iv(k_al8[:, None], alpha[None, :] / 8.0)
    
    iv_mu_al8  = I_al8[np.abs(mu_arr), :]
    iv_muN_al8 = I_al8[np.abs(mu_arr[:, None] - N_all[None, :]), :]
    
    # Costruzione Pesi Complessi
    C_s = np.concatenate([sign_s, -1j * sign_s])
    W = sign_mu[:, None, None] * C_s[None, :, None] * iv_mu_al8[:, None, :] * iv_muN_al8
    
    # L'INIEZIONE NUMBA
    temp = _numba_t_loop(t_max, G_tilde, omega, mu_arr, N_all, W, I_al)
    
    return np.conj(np.pi * temp)

def sigma_xx_Bessel(B, G_is, G_an, m_star, t_max=20, s_max=2, mu_max=7):
    return np.real(_sigma_Bessel_complex(B, G_is, G_an, m_star, t_max, s_max, mu_max))

def sigma_xy_Bessel(B, G_is, G_an, m_star, t_max=20, s_max=2, mu_max=7):
    return np.imag(_sigma_Bessel_complex(B, G_is, G_an, m_star, t_max, s_max, mu_max))

def _B_ref_bessel(G_an, m_star, alpha_ref=10.0, B_min=0.5):
    """Scelta di B_ref per sigma_0: plateau della serie (alpha(B_ref) ~ alpha_ref).
    A B_ref minore iv(k, alpha) diverge, a B_ref maggiore si lascia il plateau."""
    return max(G_an * m_star / (8.0 * K_MEV * alpha_ref), B_min)

def MR_Bessel(B, G_is, G_an, m_star):
    """MR = sigma_0/sigma_xx - 1.  sigma_0 valutato al B di plateau.
    Per B < B_ref la serie diverge: ritorna NaN (modello non definito).
    Il chiamante decide se filtrare (fit: drop NaN) o sostituire (plot: 0)."""
    B     = np.atleast_1d(B).astype(float)
    B_ref = _B_ref_bessel(G_an, m_star)
    sig0  = sigma_xx_Bessel(np.array([B_ref]), G_is, G_an, m_star)[0]
    out   = np.full_like(B, np.nan)
    mask  = B >= B_ref
    if mask.any():
        sig_stab = sigma_xx_Bessel(B[mask], G_is, G_an, m_star)
        out[mask] = sig0 / sig_stab - 1.0
    return out


# =====================================================================
# T-DIPENDENZE (factories)
# =====================================================================
def make_step_T_deps(G1_fit, G2_fit, phi_star_fit, T_exp):
    """T-dipendenza per il modello gradino:
         Gamma_i(T) = a_i * T^2   con a_i = Gamma_i_fit / T_exp^2
         phi*(T)    = PHI_MAX * tanh(T / T_scale_phi)
                      con T_scale_phi tale che phi*(T_exp) = phi_star_fit.
    Ritorna: G1_T, G2_T, phi_star_T, info (dict con a1, a2, T_scale_phi)."""
    a1 = G1_fit / T_exp**2
    a2 = G2_fit / T_exp**2
    T_scale_phi = T_exp / np.arctanh(phi_star_fit / PHI_MAX)
    def G1_T(T):       return a1 * np.asarray(T, dtype=float)**2
    def G2_T(T):       return a2 * np.asarray(T, dtype=float)**2
    def phi_star_T(T): return PHI_MAX * np.tanh(np.asarray(T, dtype=float) / T_scale_phi)
    info = dict(a1=a1, a2=a2, T_scale_phi=T_scale_phi, T_exp=T_exp)
    return G1_T, G2_T, phi_star_T, info

def make_bessel_T_deps(G_is_fit, G_an_fit, T_exp, style='ong'):
    """T-dipendenza per il modello Bessel.
    style='ong'   : G_is ~ T^2 (fononi), G_an = G_an_fit costante (cold-spots).
                    Produce picco di R_H(T) a bassa T (cuprati pseudogap).
    style='t2'    : G_is, G_an ~ T^2 (rapporto costante, no picco).
    style='linear': G_is ~ T, G_an = G_an_fit costante (rho quasi lineare in T).

    Ritorna: G_is_T, G_an_T, info (dict dei coefficienti)."""
    if style == 'ong':
        a_is = G_is_fit / T_exp**2
        def G_is_T(T): return a_is * np.asarray(T, dtype=float)**2
        def G_an_T(T): return G_an_fit * np.ones_like(np.asarray(T, dtype=float))
        info = dict(style='ong', a_is=a_is, G_an_fit=G_an_fit, T_exp=T_exp)
    elif style == 't2':
        a_is = G_is_fit / T_exp**2
        a_an = G_an_fit / T_exp**2
        def G_is_T(T): return a_is * np.asarray(T, dtype=float)**2
        def G_an_T(T): return a_an * np.asarray(T, dtype=float)**2
        info = dict(style='t2', a_is=a_is, a_an=a_an, T_exp=T_exp)
    elif style == 'linear':
        a_is = G_is_fit / T_exp
        def G_is_T(T): return a_is * np.asarray(T, dtype=float)
        def G_an_T(T): return G_an_fit * np.ones_like(np.asarray(T, dtype=float))
        info = dict(style='linear', a_is=a_is, G_an_fit=G_an_fit, T_exp=T_exp)
    else:
        raise ValueError(f'style sconosciuto: {style!r}')
    return G_is_T, G_an_T, info


# =====================================================================
# T-DIPENDENZE (forma esplicita per fit multi-T)
# =====================================================================
T_DEP_STYLES = ('linear', 'quadratic')

def eval_T_dep(T, a, c, style):
    """G(T) = a*T^p + c,  con p=1 (linear) o p=2 (quadratic)."""
    T = np.asarray(T, dtype=float)
    if style == 'linear':    return a * T    + c
    if style == 'quadratic': return a * T**2 + c
    raise ValueError(f'style sconosciuto: {style!r}  (validi: {T_DEP_STYLES})')

def fit_T_dep(T_arr, G_arr, style, maxfev=5000):
    """Fit G(T) = a*T^p + c su punti (T, G).  Ritorna (a, c)."""
    T_arr = np.asarray(T_arr, dtype=float)
    G_arr = np.asarray(G_arr, dtype=float)
    p   = 2 if style == 'quadratic' else 1
    a0  = max(1e-4, np.ptp(G_arr) / max(T_arr.max()**p, 1e-12))
    c0  = max(0.0, float(G_arr.min()))
    popt, _ = curve_fit(lambda T, a, c: eval_T_dep(T, a, c, style),
                        T_arr, G_arr, p0=[a0, c0], maxfev=maxfev)
    return float(popt[0]), float(popt[1])

def phi_tanh(T, phi_0, dphi, T_phi):
    """phi*(T) = phi_0 + dphi * tanh(T / T_phi).
    A T=0:  phi_0;  a T->inf:  phi_0 + dphi.  Radianti.
    """
    return phi_0 + dphi * np.tanh(np.asarray(T, dtype=float) / T_phi)

def fit_phi_tanh(T_arr, phi_arr, maxfev=5000):
    """Fit 3-parametri phi*(T) = phi_0 + dphi * tanh(T/T_phi).
    Vincoli: phi_0 in [0, PHI_MAX], dphi in [-PHI_MAX, PHI_MAX], T_phi > 0.
    Ritorna (phi_0, dphi, T_phi)."""
    T_arr   = np.asarray(T_arr, dtype=float)
    phi_arr = np.asarray(phi_arr, dtype=float)
    phi_0_0 = float(phi_arr[0])
    dphi_0  = float(phi_arr[-1] - phi_arr[0])
    if abs(dphi_0) < 1e-3:
        dphi_0 = 0.01
    T_phi_0 = 20.0
    bnds = ([0.0, -PHI_MAX, 0.1],
            [PHI_MAX,  PHI_MAX, 1000.0])
    popt, _ = curve_fit(phi_tanh, T_arr, phi_arr,
                        p0=[phi_0_0, dphi_0, T_phi_0],
                        bounds=bnds, maxfev=maxfev)
    return float(popt[0]), float(popt[1]), float(popt[2])


# =====================================================================
# T-DIPENDENZE — forme parametriche per uso via puntatori
# =====================================================================
import inspect as _inspect

def T_linear(T, a, c):
    r"""$a\cdot T + c$"""
    return a * np.asarray(T, float) + c

def T_quadratic(T, a, c):
    r"""$a\cdot T^2 + c$"""
    return a * np.asarray(T, float)**2 + c

def T_linear_no_offset(T, a):
    r"""$a\cdot T$"""
    return a * np.asarray(T, float)

def T_quadratic_no_offset(T, a):
    r"""a$\cdot T^2$"""
    return a * np.asarray(T, float)**2

def T_power(T, a, n, c):
    r"""$a\cdot T^n + c$"""
    return a * np.asarray(T, float)**n + c

def phi_tanh_sym(T, phi_inf, T_phi):
    r"""$\phi_\infty\,\tanh(T/T_\phi)$"""
    return phi_inf * np.tanh(np.asarray(T, float) / T_phi)

def phi_const(T, phi_0):
    r"""$\phi_0\;\text{(costante)}$"""
    return np.full_like(np.asarray(T, float), phi_0)

def T_npar(fn):
    """Numero di parametri fisici di fn(T, *params): len(params)."""
    return len(_inspect.signature(fn).parameters) - 1

def label_T_dep(fn, params, unit=''):
    """Stringa leggibile: 'prima_riga_doc  [p0, p1, ...]  unit'."""
    params_str = ', '.join(f'{v:.5g}' for v in params)
    raw = (fn.__doc__ or fn.__name__).strip()
    first_line = next((l.strip() for l in raw.splitlines() if l.strip()), raw)
    suffix = f'  {unit}' if unit else ''
    return f'{first_line}  [{params_str}]{suffix}'

def make_unpack_step(G1_fn, G2_fn, phi_fn, m_star=None):
    """Crea _unpack_step(params, T) per build_global_residual (modello gradino).

    G1_fn, G2_fn, phi_fn : callable  f(T, *params) -> scalar/array
    m_star : float | None            None = m_star libero (ultimo param del vettore).

    Attributi del risultato:
      .n_params  — lunghezza totale del vettore params
      .n1, .n2, .nphi — n. params per G1, G2, phi
      .G1_fn, .G2_fn, .phi_fn, .m_star_fixed
    """
    n1   = T_npar(G1_fn)
    n2   = T_npar(G2_fn)
    nphi = T_npar(phi_fn)

    if m_star is None:
        def _unpack(p, T):
            i = 0
            G1 = G1_fn(T, *p[i:i+n1]);    i += n1
            G2 = G2_fn(T, *p[i:i+n2]);    i += n2
            ph = phi_fn(T, *p[i:i+nphi]); i += nphi
            ms = p[i]
            return G1, G2, ph, ms
        _unpack.n_params = n1 + n2 + nphi + 1
    else:
        _ms = float(m_star)
        def _unpack(p, T):
            i = 0
            G1 = G1_fn(T, *p[i:i+n1]);    i += n1
            G2 = G2_fn(T, *p[i:i+n2]);    i += n2
            ph = phi_fn(T, *p[i:i+nphi])
            return G1, G2, ph, _ms
        _unpack.n_params = n1 + n2 + nphi

    _unpack.n1           = n1
    _unpack.n2           = n2
    _unpack.nphi         = nphi
    _unpack.G1_fn        = G1_fn
    _unpack.G2_fn        = G2_fn
    _unpack.phi_fn       = phi_fn
    _unpack.m_star_fixed = m_star
    return _unpack


# =====================================================================
# FIT MULTI-T - factory per residui globali
# =====================================================================
def log_bin(B, R, n_bins):
    """Ribinning log-uniforme in B.  Ritorna (B_bin, R_bin) con <= n_bins punti
    (meno se ci sono bin vuoti).  Media aritmetica dentro ogni bin.
    Usato per bilanciare il peso tra bassi e alti B prima di un fit."""
    B = np.asarray(B, dtype=float)
    R = np.asarray(R, dtype=float)
    if B.size == 0:
        return B, R
    B_min, B_max = float(B.min()), float(B.max())
    if B_min <= 0 or B_max <= B_min:
        return B, R
    edges  = np.logspace(np.log10(B_min), np.log10(B_max), n_bins + 1)
    idx    = np.digitize(B, edges) - 1
    idx    = np.clip(idx, 0, n_bins - 1)
    B_out, R_out = [], []
    for k in range(n_bins):
        m = idx == k
        if m.any():
            B_out.append(B[m].mean())
            R_out.append(R[m].mean())
    return np.array(B_out), np.array(R_out)


def build_global_residual(model_fn, DATA, B_MIN, unpack_fn,
                          *, n_bins=None, nan_fill=0.0,
                          b_weight_range=None, weight_factor=1.0):
    """Factory di residui per scipy.optimize.least_squares su un fit multi-T.

    Parametri Nuovi
    ---------------
    b_weight_range : tuple (float, float) | None
        Intervallo (B_start, B_end) in cui applicare un peso maggiore al fit.
    weight_factor : float
        Fattore moltiplicativo per i residui all'interno di b_weight_range.
    """
    per_T = {}
    for T, (B, R) in DATA.items():
        B = np.asarray(B, dtype=float)
        R = np.asarray(R, dtype=float)
        sel = np.isfinite(B) & np.isfinite(R) & (B > B_MIN[T])
        B_sel, R_sel = B[sel], R[sel]
        
        if n_bins is not None:
            B_sel, R_sel = log_bin(B_sel, R_sel, n_bins)
            
        # --- NOVITÀ: Calcolo dei pesi statici ---
        W_sel = np.ones_like(B_sel)
        if b_weight_range is not None:
            b_low, b_high = b_weight_range
            mask = (B_sel >= b_low) & (B_sel <= b_high)
            W_sel[mask] = weight_factor
            
        # Salviamo anche i pesi W_sel nella tupla
        per_T[T] = (B_sel, R_sel, W_sel)

    def residual(params):
        parts = []
        # Estraiamo anche W_sel ad ogni ciclo
        for T, (B_sel, R_sel, W_sel) in per_T.items():
            args  = unpack_fn(params, T)
            R_mod = np.asarray(model_fn(B_sel, *args), dtype=float)
            delta = R_mod - R_sel
            
            if not np.all(np.isfinite(delta)):
                delta = np.nan_to_num(delta, nan=nan_fill, posinf=nan_fill, neginf=nan_fill)
                
            # --- NOVITÀ: Applicazione dei pesi al residuo ---
            delta *= W_sel
            
            parts.append(delta)
        return np.concatenate(parts) if parts else np.zeros(0)

    residual.per_T = per_T
    residual.total_pts = sum(len(v[0]) for v in per_T.values())
    return residual

    def residual(params):
        parts = []
        for T, (B_sel, R_sel) in per_T.items():
            args  = unpack_fn(params, T)
            R_mod = np.asarray(model_fn(B_sel, *args), dtype=float)
            delta = R_mod - R_sel
            if not np.all(np.isfinite(delta)):
                delta = np.nan_to_num(delta, nan=nan_fill, posinf=nan_fill, neginf=nan_fill)
            parts.append(delta)
        return np.concatenate(parts) if parts else np.zeros(0)

    residual.per_T = per_T   # per introspezione (tot points, B ranges, ecc.)
    residual.total_pts = sum(len(v[0]) for v in per_T.values())
    return residual


# =====================================================================
# Utility
# =====================================================================
def clip_mr(arr, nome=''):
    """Imposta a 0 i valori negativi o NaN di MR (uso in plotting, silenzioso).
    NaN proviene tipicamente da MR_Bessel per B < B_ref.
    Il parametro `nome` e' tenuto per backward-compat ma non fa piu' print."""
    del nome  # non piu' usato
    arr = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0)
    return np.clip(arr, 0.0, None)

def load_ataei(which, data_dir='Data/Ataei'):
    """Loader generico per i file MOESM*_ESM.txt di Ataei 2022.

    Ogni file ha header commentato con #, dati separati da whitespace.
    Il numero di colonne varia: 6 per i file a 3 dataset, 4 per i file
    a 2 dataset. Il chiamante decide come re-impacchettare le colonne.
    Ritorna: np.ndarray di shape (N_righe, N_colonne).
    """
    path = os.path.join(data_dir, f'41567_2022_1763_MOESM{which}_ESM.txt')
    arr = pd.read_csv(path, sep=r'\s+', comment='#', header=None).to_numpy()
    return arr

def load_ataei_extended(which_list, data_dir='Data/Ataei'):
    """
    Carica i dati da uno o più file MOESM e li concatena orizzontalmente.
    which_list: può essere un intero (es. 2) o una lista (es. [2, 6])
    """
    if isinstance(which_list, int):
        which_list = [which_list]
    
    all_data = []
    for which in which_list:
        path = os.path.join(data_dir, f'41567_2022_1763_MOESM{which}_ESM.txt')
        # Usiamo sep=r'\s+' per gestire tab o spazi multipli
        df = pd.read_csv(path, sep=r'\s+', comment='#', header=None)
        all_data.append(df)
    
    # Concateniamo le colonne (axis=1)
    # Nota: se i file hanno lunghezze diverse, pandas aggiungerà dei NaN automaticamente
    combined_df = pd.concat(all_data, axis=1)
    return combined_df.to_numpy()


if __name__ == '__main__':
    print(__doc__)
