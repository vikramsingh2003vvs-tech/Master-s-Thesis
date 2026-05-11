import json, uuid, io

path = r'C:/Users/vikra/OneDrive/Documenti/File/Uni/Magistrale/Tesi/jupyter tesi/Magnetoresistence.ipynb'
with io.open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md_cell(src):
    return {"cell_type":"markdown","id":str(uuid.uuid4()),"metadata":{},"source":src.splitlines(keepends=True)}

def code_cell(src):
    return {"cell_type":"code","id":str(uuid.uuid4()),"metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}

# =========================================================================
# 1) Modifica cella delle funzioni: aggiungi _sigma_complex_step e sigma_xy_step
# =========================================================================
cell_fn = next(c for c in nb['cells'] if c.get('id') == 'a0000004')
src = ''.join(cell_fn['source'])

old = '''def sigma_xx_step(B, Gamma1, Gamma2, phi_star, m_star):
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
    return np.real(O_I1 + O_I2)'''

new = '''def _sigma_complex_step(B, Gamma1, Gamma2, phi_star, m_star):
    """Ritorna O_I1 + O_I2 complesso.
       sigma_xx = Re(.)   sigma_xy = Im(.)"""
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
    return O_I1 + O_I2

def sigma_xx_step(B, Gamma1, Gamma2, phi_star, m_star):
    return np.real(_sigma_complex_step(B, Gamma1, Gamma2, phi_star, m_star))

def sigma_xy_step(B, Gamma1, Gamma2, phi_star, m_star):
    return np.imag(_sigma_complex_step(B, Gamma1, Gamma2, phi_star, m_star))'''

assert old in src, 'sigma_xx_step originale non trovata'
src = src.replace(old, new)
cell_fn['source'] = src.splitlines(keepends=True)

# =========================================================================
# 2) Aggiorna markdown della sezione T
# =========================================================================
cell_md = next(c for c in nb['cells'] if c.get('id') == 'dfc4001a-10ed-4b9d-abb0-69038d2cc405')
cell_md['source'] = r"""---
## Dipendenza dalla temperatura (modello a GRADINO)
Inseriamo una dipendenza da $T$ nei parametri ottenuti dal fit:

- $\Gamma_i(T) = a_i\,T^2 + b_i$
- $\phi^*(T) = c\,T + \phi^*_0$

I coefficienti $b_1, b_2, \phi^*_0$ sono i valori del fit (interpretati come $T\to 0$).
Modifica $a_1, a_2, c$ per cambiare la dipendenza termica.

Su questa griglia $(B, T)$ si calcolano:
- $\rho_{xx}(B, T) = 1/\sigma_{xx}$
- $R_H(B, T) = \sigma_{xx}/(B\,\sigma_{xy})$
- $\cot\theta_H(B, T) = \sigma_{xx}/\sigma_{xy}$

con $\sigma_{xy} = \mathrm{Im}(O_{I_1} + O_{I_2})$.
""".splitlines(keepends=True)

# =========================================================================
# 3) Sostituisci la cella di calcolo con una che produce tutte le griglie + helper
# =========================================================================
cell_compute = next(c for c in nb['cells'] if c.get('id') == '4348456f-3509-455b-8e58-7aee25136e61')

new_compute = r'''# =================================================================
# Griglie (B, T) per sigma_xx, sigma_xy, rho_xx, R_H, cot(theta_H)
# =================================================================

# Valori di riferimento dal fit (T -> 0)
G1_0, G2_0, phi_star_0, m_star_T = popt_step
print(f'Valori dal fit: G1_0={G1_0:.3f} meV, G2_0={G2_0:.3f} meV, '
      f'phi*_0={np.degrees(phi_star_0):.2f} deg, m*={m_star_T:.3f} m_e')

# -- Coefficienti della dipendenza da T (modificabili) ---------------
a1    = 0.004              # Gamma1(T) = a1*T^2 + G1_0           [meV/K^2]
a2    = 0.010              # Gamma2(T) = a2*T^2 + G2_0           [meV/K^2]
c_phi = np.radians(0.10)   # phi*(T)   = c_phi*T + phi*_0        [rad/K]

def G1_T(T):     return a1 * T**2 + G1_0
def G2_T(T):     return a2 * T**2 + G2_0
def phi_star_T(T):
    return np.clip(c_phi * T + phi_star_0, 1e-4, np.pi/4 - 1e-4)

# Griglia (B, T)
B_grid_1d = np.linspace(1.0, 90.0, 80)
T_grid_1d = np.linspace(0.0, 100.0, 60)
B_mesh, T_mesh = np.meshgrid(B_grid_1d, T_grid_1d)

sigxx_grid = np.zeros_like(B_mesh)
sigxy_grid = np.zeros_like(B_mesh)
for i, T in enumerate(T_grid_1d):
    sig = _sigma_complex_step(B_grid_1d, G1_T(T), G2_T(T), phi_star_T(T), m_star_T)
    sigxx_grid[i, :] = np.real(sig)
    sigxy_grid[i, :] = np.imag(sig)

# Quantita derivate
rho_xx_grid = 1.0 / sigxx_grid
R_H_grid    = sigxx_grid / (B_mesh * sigxy_grid)
cot_H_grid  = sigxx_grid / sigxy_grid

print(f'rho_xx   in [{rho_xx_grid.min():.3g}, {rho_xx_grid.max():.3g}]')
print(f'R_H      in [{R_H_grid.min():.3g}, {R_H_grid.max():.3g}]')
print(f'cot(θH)  in [{cot_H_grid.min():.3g}, {cot_H_grid.max():.3g}]')


# =================================================================
# Helper: 4 plot (plotly surface + plotly contour + matplotlib B/T slices)
# =================================================================
def plot_observable_4panel(Z_grid, name, label_plain, label_tex, cmap='Viridis',
                           T_slices=(0, 10, 20, 30, 40, 50),
                           B_slices=(5, 20, 40, 60, 80)):
    """Produce i 4 plot standard di una osservabile Z(B, T).

    Parameters
    ----------
    Z_grid      : array (N_T, N_B) valutato sulla griglia (B_grid_1d, T_grid_1d)
    name        : es. 'rho_xx'        (usato nei titoli)
    label_plain : es. 'ρ_xx'          (hover/colorbar plotly)
    label_tex   : es. r'$\rho_{xx}$'  (ylabel matplotlib)
    cmap        : colorscale plotly
    """
    # ---- Plotly: 3D surface + contour 2D ----------------------------
    surface = go.Surface(
        x=B_grid_1d, y=T_grid_1d, z=Z_grid,
        colorscale=cmap,
        colorbar=dict(title=dict(text=label_plain), x=0.45, len=0.85),
        hovertemplate=f'B = %{{x:.1f}} T<br>T = %{{y:.1f}} K<br>{label_plain} = %{{z:.4g}}<extra></extra>',
        contours=dict(z=dict(show=True, usecolormap=True,
                             highlightcolor='white', project_z=True)),
    )
    contour = go.Contour(
        x=B_grid_1d, y=T_grid_1d, z=Z_grid,
        colorscale=cmap,
        colorbar=dict(title=dict(text=label_plain), x=1.02, len=0.85),
        contours=dict(showlabels=True, labelfont=dict(size=11, color='white')),
        hovertemplate=f'B = %{{x:.1f}} T<br>T = %{{y:.1f}} K<br>{label_plain} = %{{z:.4g}}<extra></extra>',
        line=dict(width=0.5, color='rgba(255,255,255,0.4)'),
    )
    fig_p = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'xy'}]],
        subplot_titles=(f'Superficie {name}(B, T)', f'Contour {name}(B, T)'),
        horizontal_spacing=0.15,
    )
    fig_p.add_trace(surface, row=1, col=1)
    fig_p.add_trace(contour, row=1, col=2)
    fig_p.update_scenes(xaxis_title='B (T)', yaxis_title='T (K)', zaxis_title=label_plain,
                        camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)), row=1, col=1)
    fig_p.update_xaxes(title_text='B (T)', row=1, col=2)
    fig_p.update_yaxes(title_text='T (K)', row=1, col=2)
    fig_p.update_layout(
        height=550, width=1200,
        title=f'{name}(B, T) - gradino',
        margin=dict(l=10, r=10, t=80, b=10),
    )
    fig_p.show()

    # ---- Matplotlib: Z(B) a T fissa  +  Z(T) a B fissa --------------
    fig, (axB, axT) = plt.subplots(1, 2, figsize=(14, 5))

    col_T = cm.plasma(np.linspace(0.05, 0.85, len(T_slices)))
    for Ts, col in zip(T_slices, col_T):
        i = int(np.argmin(np.abs(T_grid_1d - Ts)))
        axB.plot(B_grid_1d, Z_grid[i, :], color=col, linewidth=2, label=f'T = {Ts} K')
    axB.set_xlabel('B  (T)', fontsize=12)
    axB.set_ylabel(label_tex, fontsize=12)
    axB.set_title(f'{name}(B) a T fissa', fontsize=13)
    axB.legend(fontsize=9); axB.grid(linestyle='--', alpha=0.5)

    col_B = cm.viridis(np.linspace(0.15, 0.9, len(B_slices)))
    for Bs, col in zip(B_slices, col_B):
        j = int(np.argmin(np.abs(B_grid_1d - Bs)))
        axT.plot(T_grid_1d, Z_grid[:, j], color=col, linewidth=2, label=f'B = {Bs} T')
    axT.set_xlabel('T  (K)', fontsize=12)
    axT.set_ylabel(label_tex, fontsize=12)
    axT.set_title(f'{name}(T) a B fissa', fontsize=13)
    axT.legend(fontsize=9); axT.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

print('Griglie e helper pronti.')
'''
cell_compute['source'] = new_compute.splitlines(keepends=True)

# =========================================================================
# 4) Sostituisci la cella dei 4 plot con la chiamata per rho_xx
# =========================================================================
cell_plot = next(c for c in nb['cells'] if c.get('id') == 'ef42245c-ad17-4ea4-8dec-2e44f3ba9dd6')
cell_plot['source'] = r'''plot_observable_4panel(
    rho_xx_grid,
    name='ρ_xx',
    label_plain='ρ_xx',
    label_tex=r'$\rho_{xx}$',
)
'''.splitlines(keepends=True)

# =========================================================================
# 5) Rimuovi la cella vuota finale se presente
# =========================================================================
nb['cells'] = [c for c in nb['cells']
               if not (c['cell_type'] == 'code' and not ''.join(c['source']).strip())]

# =========================================================================
# 6) Aggiungi sezioni R_H e cot(theta_H)
# =========================================================================
md_RH = r"""---
## Coefficiente di Hall $R_H(B, T)$

$R_H = \dfrac{\sigma_{xx}}{B\,\sigma_{xy}}$
"""
code_RH = r'''plot_observable_4panel(
    R_H_grid,
    name='R_H',
    label_plain='R_H',
    label_tex=r'$R_H$',
    cmap='Cividis',
)
'''

md_cot = r"""---
## Cotangente dell'angolo di Hall $\cot\theta_H(B, T)$

$\cot\theta_H = \dfrac{\sigma_{xx}}{\sigma_{xy}}$
"""
code_cot = r'''plot_observable_4panel(
    cot_H_grid,
    name='cot(θ_H)',
    label_plain='cot(θ_H)',
    label_tex=r'$\cot\theta_H$',
    cmap='Plasma',
)
'''

nb['cells'].extend([md_cell(md_RH), code_cell(code_RH),
                    md_cell(md_cot), code_cell(code_cot)])

with io.open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('OK, totale celle =', len(nb['cells']))
