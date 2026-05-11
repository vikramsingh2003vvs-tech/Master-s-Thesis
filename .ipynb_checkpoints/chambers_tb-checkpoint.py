"""
chambers_tb.py  —  Magnetoresistenza numerica di Chambers con FS tight-binding 2D.

Modello di scattering anisotropo (come Mirarchi & Caprara, A153, 2024):
    1/tau(phi_k) = G_is + G_an * |cos(2*phi_k)|^nu     [meV]

dove phi_k è l'angolo di k sulla FS (0 = direzione (pi,0), pi/4 = nodale).

Dispersione tight-binding tetragonale:
    eps(kx, ky) = -2t*(cos kx + cos ky)
                  -4t'*cos kx*cos ky
                  -2t''*(cos 2kx + cos 2ky)
                  - mu

Unità naturali: hbar = 1, a = 1 (reticolo).
  k adimensionale in [-pi, pi]
  energie in meV
  velocità vF = d eps/dk  in meV
  tempo in 1/meV

Formula di Chambers per sigma_xx in 2D (derivazione in Ziman 1960):

  sigma_xx(B) ∝ ∮_FS (ds / v_F) * v_x(k0) * J_x(k0, B)

  J_x(k0, B) = ∫_0^inf v_x(k(-s)) * exp(-∫_0^s gamma(k) / (KMEV*B*vF) ds') * ds/(KMEV*B*vF)

dove il cammino k(-s) va all'indietro lungo l'orbita ciclotronica
(equivalente a integrare 'backward in time').

MR(B) = sigma_xx(B=eps) / sigma_xx(B) - 1
"""

import numpy as np
from scipy.interpolate import interp1d
from numba import njit, prange

# ─── costante fisica (da chambers_lib) ────────────────────────────────────
K_MEV = 2.17e-4   # hbar * e / m_e  [meV / T]

# ─── parametri TB default: Nd-LSCO p=0.24, Mirarchi A153 ─────────────────
TB_ND_LSCO = dict(t=435.0, tp=-50.0, tpp=38.0, p=0.24, nu=4)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Tight-binding dispersion e gradiente
# ═══════════════════════════════════════════════════════════════════════════

def tb_dispersion(kx, ky, t, tp, tpp, mu):
    return (-2*t*(np.cos(kx) + np.cos(ky))
            - 4*tp*np.cos(kx)*np.cos(ky)
            - 2*tpp*(np.cos(2*kx) + np.cos(2*ky))
            - mu)


def tb_velocity(kx, ky, t, tp, tpp):
    """vx = d eps/d kx,  vy = d eps/d ky  [meV, con a=1]"""
    vx = (2*t*np.sin(kx)
          + 4*tp*np.sin(kx)*np.cos(ky)
          + 4*tpp*np.sin(2*kx))
    vy = (2*t*np.sin(ky)
          + 4*tp*np.cos(kx)*np.sin(ky)
          + 4*tpp*np.sin(2*ky))
    return vx, vy


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Determinazione di mu per doping p
# ═══════════════════════════════════════════════════════════════════════════

def find_mu(p, t, tp, tpp, N_grid=512, n_iter=60, verbose=False):
    """
    Trova mu tale che il riempimento della banda (spin incluso) = 1 - p.
    Convenzione hole-doped: filling = 1 - p  (meta' banda = filling 0.5 -> p=0.5).

    Usa il conteggio di stati sul reticolo k (metodo count).
    verbose=True: mostra una barra tqdm per le iterazioni di bisection.
    """
    from tqdm.auto import tqdm as _tqdm

    kx = np.linspace(-np.pi, np.pi, N_grid, endpoint=False)
    ky = np.linspace(-np.pi, np.pi, N_grid, endpoint=False)
    KX, KY = np.meshgrid(kx, ky)
    N_tot = N_grid**2

    target_filling = (1.0 - p) / 2.0

    def filling_at_mu(mu):
        eps = tb_dispersion(KX, KY, t, tp, tpp, mu)
        return np.sum(eps < 0) / N_tot   # stati con eps < 0 sono occupati

    # Bisection su mu
    mu_lo, mu_hi = -4*np.abs(t), 4*np.abs(t)
    iters = _tqdm(range(n_iter), desc='bisection mu', leave=False,
                  unit='it') if verbose else range(n_iter)
    for _ in iters:
        mu_mid = 0.5*(mu_lo + mu_hi)
        if filling_at_mu(mu_mid) < target_filling:
            mu_lo = mu_mid
        else:
            mu_hi = mu_mid
        if verbose:
            iters.set_postfix({'mu': f'{mu_mid:.1f} meV'})
    return 0.5*(mu_lo + mu_hi)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Estrazione del contorno della FS
# ═══════════════════════════════════════════════════════════════════════════

def get_fermi_surface(t, tp, tpp, mu, N_grid=400, N_interp=512):
    import matplotlib.pyplot as plt

    # 1. Griglia da 0 a 2*pi. In questo modo la tasca di lacune è 
    # perfettamente centrata in (pi, pi) e non tocca MAI i bordi.
    kx = np.linspace(0, 2*np.pi, N_grid)
    ky = np.linspace(0, 2*np.pi, N_grid)
    KX, KY = np.meshgrid(kx, ky)
    EPS = tb_dispersion(KX, KY, t, tp, tpp, mu)

    fig, ax = plt.subplots()
    cs = ax.contour(KX, KY, EPS, levels=[0.0])
    plt.close(fig)

    # 2. Estrai il contorno (ora sarà un cerchio/ovale chiuso perfetto)
    contours = [p.vertices for p in cs.get_paths()]
    contour = max(contours, key=len)
    kx_raw = contour[:, 0].copy()
    ky_raw = contour[:, 1].copy()

    # Assicuriamoci che il loop sia chiuso per l'interpolazione
    if (kx_raw[0] != kx_raw[-1]) or (ky_raw[0] != ky_raw[-1]):
        kx_raw = np.append(kx_raw, kx_raw[0])
        ky_raw = np.append(ky_raw, ky_raw[0])

    # 3. Interpolazione rispetto all'arco (su coordinate continue, niente salti!)
    ds_raw = np.sqrt(np.diff(kx_raw)**2 + np.diff(ky_raw)**2)
    s_raw = np.concatenate([[0], np.cumsum(ds_raw)])
    s_total = s_raw[-1]
    s_new = np.linspace(0, s_total, N_interp, endpoint=False)

    kx_fs = interp1d(s_raw, kx_raw, kind='linear')(s_new)
    ky_fs = interp1d(s_raw, ky_raw, kind='linear')(s_new)

    # 4. Velocità (sono periodiche, funzionano anche in coordinate [0, 2pi])
    vx_fs, vy_fs = tb_velocity(kx_fs, ky_fs, t, tp, tpp)

    # 5. Angolo phi rispetto a Gamma(0,0) come in A153.
    # Dobbiamo riportare i k in [-pi, pi] per far funzionare bene l'arctan2
    kx_gamma = (kx_fs + np.pi) % (2 * np.pi) - np.pi
    ky_gamma = (ky_fs + np.pi) % (2 * np.pi) - np.pi
    phi_fs = np.arctan2(ky_gamma, kx_gamma)

    # Elementi di arco (distanza tra punti vicini)
    dkx = np.roll(kx_fs, -1) - np.roll(kx_fs, 1)
    dky = np.roll(ky_fs, -1) - np.roll(ky_fs, 1)
    ds_fs = 0.5 * np.sqrt(dkx**2 + dky**2)

    return kx_fs, ky_fs, vx_fs, vy_fs, phi_fs, ds_fs
# ═══════════════════════════════════════════════════════════════════════════
# 4.  Tasso di scattering sulla FS
# ═══════════════════════════════════════════════════════════════════════════

def scattering_rate(phi_fs, G_is, G_an, nu=4):
    """
    gamma(phi) = G_is + G_an * |cos(2*phi)|^nu   [meV]

    phi = 0 : antinodo (pi, 0)  — hot spot
    phi = pi/4: nodo (pi/2, pi/2) — cold spot
    """
    return G_is + G_an * np.abs(np.cos(2.0 * phi_fs))**nu


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Calcolo di sigma_xx con la formula di Chambers (orbita discreta)
# ═══════════════════════════════════════════════════════════════════════════

def _chambers_sigma(B, vx_fs, vy_fs, gamma_fs, ds_fs, N_orbit_max=12, tol=1e-8):
    """
    Calcola sigma_xx(B) in unità interne (A si cancella nell MR).

    L'orbita va all'indietro (backward in time): dalla particella che si muove
    nel verso opposto alla frequenza ciclotrone.

    Per un foro (hole-like FS), la direzione dell'orbita è opposta; siccome
    calcoliamo solo il rapporto sigma_xx(0)/sigma_xx(B) la scelta del verso
    non modifica MR.
    """
    N = len(vx_fs)
    vF_fs = np.sqrt(vx_fs**2 + vy_fs**2)

    if B <= 0:
        # Limite B=0: J_x(k0) = vx(k0)/gamma(k0)
        w = ds_fs / vF_fs          # peso geometrico dl/v_F
        return float(np.sum(w * vx_fs * vx_fs / gamma_fs))

    # Elemento di tempo (in unità 1/meV):  dt_j = ds_j / (K_MEV * B * v_F_j)
    dt_fs = ds_fs / (K_MEV * B * vF_fs)

    # Damping totale per un'orbita completa
    gamma_total = float(np.sum(gamma_fs * dt_fs))

    # Numero di orbite necessarie: N_orbit s.t. exp(-n*gamma_total) < tol
    if gamma_total > 0:
        n_max = min(N_orbit_max, int(np.ceil(-np.log(tol) / gamma_total)) + 1)
    else:
        n_max = N_orbit_max

    # Pre-calcola il fattore di attenuazione per un'orbita completa
    exp_one_orbit = np.exp(-gamma_total)

    sigma = 0.0
    for i0 in range(N):
        J_x = 0.0
        cum_damp = 0.0

        for n in range(n_max):
            orbit_factor = exp_one_orbit**n
            if orbit_factor < tol:
                break
            for step in range(N):
                j = (i0 - step) % N   # indice backward
                exp_fac = orbit_factor * np.exp(-cum_damp)
                J_x += vx_fs[j] * exp_fac * dt_fs[j]
                cum_damp += gamma_fs[j] * dt_fs[j]
            # dopo il primo giro, cum_damp = gamma_total * (n+1), ma lo usiamo già

        sigma += (ds_fs[i0] / vF_fs[i0]) * vx_fs[i0] * J_x

    return sigma


def _chambers_sigma_fast(B, vx_fs, vy_fs, gamma_fs, ds_fs, N_orbit_max=12, tol=1e-8):
    """
    Versione vettorizzata (più veloce) di _chambers_sigma.
    Costruisce la matrice di attenuazione exp(-integral gamma dt) dal punto
    i0 al punto j, poi accumula con numpy.
    """
    N = len(vx_fs)
    vF_fs = np.sqrt(vx_fs**2 + vy_fs**2)

    if B <= 0:
        w = ds_fs / vF_fs
        return float(np.sum(w * vx_fs**2 / gamma_fs))

    dt_fs = ds_fs / (K_MEV * B * vF_fs)
    gamma_dt = gamma_fs * dt_fs   # attenuazione per step

    gamma_total = float(gamma_dt.sum())

    if gamma_total > 0:
        n_max = min(N_orbit_max, int(np.ceil(-np.log(tol) / gamma_total)) + 1)
    else:
        n_max = N_orbit_max

    # Cumulative damping backward: cumsum_back[j] = sum_{step=0}^{j-1} gamma_dt[i0-step]
    # Precalcolo: cumsum backward da ogni punto i0.
    # Per efficienza, uso il fatto che l'orbita è periodica: ruotiamo gamma_dt.
    # cumsum_back[i0, j] = sum_{s=0}^{j-1} gamma_dt[(i0-s) % N]
    # Equivalente a: cumsum della sequenza gamma_dt rovesciata e ruotata.

    # gamma_dt_rev[j] = gamma_dt[-j % N] = gamma_dt[(N-j) % N]
    gamma_dt_rev = gamma_dt[::-1]   # inversione (equivale a reverse orbit)

    # Per ogni i0, la sequenza backward è: gamma_dt[i0], gamma_dt[i0-1], ...
    # il cumsum è: cs[s] = sum_{j=0}^{s-1} gamma_dt[(i0-j)%N]
    # Usiamo np.roll per spostare e poi cumsum

    # Precomputa: cs_full[i0, s] = cumulative damp dal punto i0 per s passi backward
    # Questo è O(N^2) memoria; per N=512 è 512^2 float64 = 2MB, accettabile.

    # Primo giro (n=0): cs_full[i0, s] = sum_{j=0}^{s-1} gamma_dt[(i0-j)%N]
    # = sum_{j=0}^{s-1} gamma_dt_rev[(N-i0+j) % N]  ... complicato.

    # Approccio semplice: costruiamo la matrice gamma_path[i0, s]
    # dove s=0..N-1 è il passo backward, e (i0-s)%N è l'indice FS.
    idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N   # shape (N, N)
    # idx[i0, s] = (i0 - s) % N

    gamma_path = gamma_dt[idx]    # gamma_path[i0, s] = gamma_dt[(i0-s)%N]
    vx_path    = vx_fs[idx]       # vx[(i0-s)%N]
    dt_path    = dt_fs[idx]       # dt[(i0-s)%N]

    # Cumulative damp per ogni i0: cs[i0, s] = sum_{j=0}^{s-1} gamma_path[i0, j]
    cs = np.zeros((N, N))
    cs[:, 1:] = np.cumsum(gamma_path[:, :-1], axis=1)

    exp_fac_1orbit = np.exp(-cs)   # shape (N, N), primo giro

    # J_x per il primo giro
    J_x = np.sum(vx_path * exp_fac_1orbit * dt_path, axis=1)   # shape (N,)

    # Orbite successive: ogni n-esima orbita contribuisce exp(-n*gamma_total) * J_x_1orbit
    exp_one = np.exp(-gamma_total)
    factor = 1.0
    for n in range(1, n_max):
        factor *= exp_one
        if factor < tol:
            break
        J_x += factor * np.sum(vx_path * exp_fac_1orbit * dt_path, axis=1)

    # sigma_xx = sum_{i0} (ds/vF)[i0] * vx[i0] * J_x[i0]
    sigma = float(np.sum((ds_fs / vF_fs) * vx_fs * J_x))
    return sigma


# ═══════════════════════════════════════════════════════════════════════════
# 6.  MR pubblica
# ═══════════════════════════════════════════════════════════════════════════

def MR_chambers_tb(B_arr, G_is, G_an,
                   t=435.0, tp=-50.0, tpp=38.0, p=0.24, nu=4,
                   mu=None, N_fs=512, N_orbit_max=12,
                   _cache={}):
    """
    MR(B) = sigma_xx(0) / sigma_xx(B) - 1  con la FS tight-binding numerica.

    Parametri
    ---------
    B_arr   : array di B [T]
    G_is    : rate isotropo [meV]
    G_an    : rate anisotropo [meV]
    t, tp, tpp : hopping [meV]  (default: Nd-LSCO p=0.24 da Mirarchi A153)
    p       : doping (usato per trovare mu)
    nu      : esponente anisotropia (default 4)
    mu      : potenziale chimico [meV]; se None viene calcolato da p
    N_fs    : punti sulla FS
    N_orbit_max: orbite massime nell'integrale di Chambers

    Ritorna
    -------
    MR_arr : array della stessa shape di B_arr
    """
    B_arr = np.atleast_1d(np.asarray(B_arr, dtype=float))

    # Chiave cache per la geometria della FS (indipendente da G_is, G_an, B)
    geom_key = (round(t,1), round(tp,1), round(tpp,1), round(p,4), N_fs)
    if geom_key not in _cache:
        if mu is None:
            mu = find_mu(p, t, tp, tpp, N_grid=600)
        kx_fs, ky_fs, vx_fs, vy_fs, phi_fs, ds_fs = get_fermi_surface(
            t, tp, tpp, mu, N_grid=400, N_interp=N_fs
        )
        _cache[geom_key] = (mu, kx_fs, ky_fs, vx_fs, vy_fs, phi_fs, ds_fs)
    else:
        mu, kx_fs, ky_fs, vx_fs, vy_fs, phi_fs, ds_fs = _cache[geom_key]

    gamma_fs = scattering_rate(phi_fs, G_is, G_an, nu)

    # sigma_xx a B~0 (limite analitico)
    vF_fs = np.sqrt(vx_fs**2 + vy_fs**2)
    sig0 = float(np.sum((ds_fs / vF_fs) * vx_fs**2 / gamma_fs))

    MR_arr = np.empty_like(B_arr)
    for i, B in enumerate(B_arr):
        if B < 0.1:
            MR_arr[i] = 0.0
            continue
        sig_B = _chambers_sigma_fast(B, vx_fs, vy_fs, gamma_fs, ds_fs,
                                     N_orbit_max=N_orbit_max)
        if sig_B > 0 and np.isfinite(sig_B):
            MR_arr[i] = sig0 / sig_B - 1.0
        else:
            MR_arr[i] = np.nan

    return MR_arr


def MR_chambers_tb_single(B, G_is, G_an, **kwargs):
    """Wrapper scalare per curve_fit."""
    return float(MR_chambers_tb(np.array([float(B)]), G_is, G_an, **kwargs)[0])

#══════════════════════════════════════════════════════════════════════════════
# 7 Numba optimization
#══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# 7. Motore di Ottimizzazione Chambers (Numba + Python)
# ═══════════════════════════════════════════════════════════════════════════

@njit(parallel=True)
def _compute_mr_core(B_arr, G_is, G_an, cos2phi_nu, sig0_kernel, dt_path_1, 
                     vx_path, weight, gt_is, gt_an, cs_1_is, cs_1_an, N):
    
    gamma_fs = G_is + G_an * cos2phi_nu
    sig0 = np.sum(sig0_kernel / gamma_fs)
    cs_1 = G_is * cs_1_is + G_an * cs_1_an
    gamma_total_1 = G_is * gt_is + G_an * gt_an
    
    MR_arr = np.zeros_like(B_arr)
    
    gamma_path = np.zeros((N, N))
    
    # Questo ciclo si può parallelizzare facilmente
    for i in prange(N):  # <-- 3. Uso prange al posto di range
        for j in range(N):
            gamma_path[i, j] = gamma_fs[(i - j) % N]
    
    for b_idx in range(len(B_arr)):
        B = B_arr[b_idx]
        if B < 0.5:
            continue
            
        J_x_1orbit = np.zeros(N)
        
        # IL VERO COLLO DI BOTTIGLIA È QUI: PARALLELIZZIAMOLO!
        for i in prange(N):  # <-- 3. I core si dividono il calcolo dell'integrale
            for j in range(N):
                g_safe = max(gamma_path[i, j], 1e-12)
                exp_term = np.exp(-cs_1[i, j] / B)
                step_int = (1.0 - np.exp(-g_safe * dt_path_1[i, j] / B)) / g_safe
                J_x_1orbit[i] += vx_path[i, j] * exp_term * step_int
        
        denom = 1.0 - np.exp(-gamma_total_1 / B)
        multiorbit = 1.0 / denom if denom > 1e-9 else B / max(gamma_total_1, 1e-30)
        
        sigma_B = 0.0
        for i in range(N):
            sigma_B += J_x_1orbit[i] * multiorbit * weight[i]
            
        if sigma_B > 0:
            MR_arr[b_idx] = sig0 / sigma_B - 1.0
        else:
            MR_arr[b_idx] = np.nan
            
    return MR_arr

import time # Serve per i print della classe

class ChambersFitEngine:
    """
    Gestisce il setup della Superficie di Fermi usando il modello Tight-Binding
    e calcola la Magnetoresistenza passando gli array al motore Numba.
    """
    def __init__(self, t=435., tp=-50., tpp=38., p=0.24, nu=4,
                 N_fs=256, N_grid_fs=400, N_grid_mu=600, n_bisect=60):
        self.nu = nu
        self.N  = N_fs

        print('find_mu...', end=' ', flush=True)
        t0 = time.time()
        # Nota: Qui usi la funzione 'find_mu' definita nello stesso file!
        mu = find_mu(p, t, tp, tpp, N_grid=N_grid_mu, n_iter=n_bisect)
        self.mu = mu
        print(f'mu={mu:.1f} meV  ({time.time()-t0:.1f}s)')

        print('Fermi surface...', end=' ', flush=True)
        t0 = time.time()
        # Anche 'get_fermi_surface' è nello stesso file
        kx, ky, vx, vy, phi, ds = get_fermi_surface(
            t, tp, tpp, mu, N_grid=N_grid_fs, N_interp=N_fs)
        print(f'N_fs={N_fs}  ({time.time()-t0:.2f}s)')

        self.kx_fs = kx; self.ky_fs = ky
        self.vx_fs = vx; self.vy_fs = vy
        self.phi_fs = phi; self.ds_fs = ds
        self.vF_fs  = np.sqrt(vx**2 + vy**2)
        N = N_fs
        self.cos2phi_nu  = np.abs(np.cos(2.0 * phi))**nu
        self.sig0_kernel = (ds / self.vF_fs) * vx**2
        self.weight      = (ds / self.vF_fs) * vx

        print('percorsi backward...', end=' ', flush=True)
        t0 = time.time()
        idx            = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
        self.vx_path   = vx[idx]
        # K_MEV è definito all'inizio di chambers_tb.py
        dt_fs_1        = ds / (K_MEV * 1.0 * self.vF_fs) 
        self.dt_path_1 = dt_fs_1[idx]
        print(f'({time.time()-t0:.2f}s)')

        print('cumsum A_is, A_an...', end=' ', flush=True)
        t0 = time.time()
        a1 = dt_fs_1
        a2 = self.cos2phi_nu * dt_fs_1
        a1_path = a1[idx]
        a2_path = a2[idx]
        self.cs_1_is = np.zeros((N, N))
        self.cs_1_is[:, 1:] = np.cumsum(a1_path[:, :-1], axis=1)
        self.cs_1_an = np.zeros((N, N))
        self.cs_1_an[:, 1:] = np.cumsum(a2_path[:, :-1], axis=1)
        self.gt_is = float(a1.sum())
        self.gt_an = float(a2.sum())
        print(f'({time.time()-t0:.2f}s)')

        print(f'Engine pronto: N_fs={N_fs}, mu={mu:.1f} meV, perimetro FS={ds.sum():.4f}')

    def compute_MR(self, B_arr, G_is, G_an, B_thr=0.5):
        """
        Interfaccia Python che passa gli array nudi alla funzione superveloce in Numba.
        """
        B_arr = np.atleast_1d(np.asarray(B_arr, dtype=float))
        
        # Chiamata al core compilato C++
        return _compute_mr_core(
            B_arr, G_is, G_an, self.cos2phi_nu, self.sig0_kernel, 
            self.dt_path_1, self.vx_path, self.weight, self.gt_is, 
            self.gt_an, self.cs_1_is, self.cs_1_an, self.N
        )