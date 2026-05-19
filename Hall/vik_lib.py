import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.stats import jackknife_resampling
from astropy.stats import jackknife_stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import sympy as sp
from sympy import symbols
from matplotlib.ticker import FuncFormatter
from fractions import Fraction
from tqdm.notebook import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import cm

font1 = {'family': 'serif', 'color': 'black', 'size': 20}
font2 = {'family': 'serif', 'color': 'black', 'size': 15}

def var_y_cov(function_sympy, data, *args, **kwargs):
    key = []
    value = []
    count = 0
    for key_old, value_old in kwargs.items():
        key.append(key_old)
        value.append(value_old)
    n = len(data)
    cov_matrix = value[0]
    Y = function_sympy
    x_ = symbols('x0:%d' % len(Y.free_symbols))
    var_y = np.zeros(n)
    temp_y = np.zeros(n)
    for i in range(len(x_) - 1):
        for j in range(len(x_) - 1):
            formula = Y.diff(x_[i + 1]) * Y.diff(x_[j + 1])
            temp = sp.lambdify(x_, formula, "numpy")
            try:
                temp_uy = temp(data, *args)
                var_y += temp_uy * cov_matrix[i][j]
                uy = np.sqrt(np.abs(var_y))
            except NameError as e:
                print("NameError: ", e)
                print("\nIl compilatore non è riuscito a eseguire le derivate, si usa quindi il metodo del jackknife.\n")
                print("Poiché è stato usato il metodo del jaccknife, si controlli che i dati inseriti siano dati sperimenatli")
                func = sp.lambdify(x_, Y, "numpy")
                x_mean = np.mean(data)
                x_jk = x_mean + (x_mean - data) / (n - 1)
                y_jk = func(x_jk, *args)
                y_jk_mean = np.sum(y_jk) / n
                y_mean = func(x_mean, *args)
                jk_std = ((n - 1) / n) * np.sum((y_mean - y_jk_mean)**2)
                uy = np.repeat(np.sqrt(jk_std),n)
                temp_y = sp.lambdify(x_, Y, "numpy")
                y = temp_y(data, *args)
                for j in range(n):
                    if (uy[0] <= y[j]/100):
                        count += 1
                if (count >= int(n/2)):
                    for i in range(len(x_) - 1):
                        var_y += cov_matrix[i][i] * ((y / args[i])**2)
                        uy = np.sqrt(var_y)
    return(uy)


def fit_jk(fun, x, y, sigma_y, guess):
    params_jackknife = []
    pcov_jackknife = []
    
    for i in range(len(x)):
        X_jack = np.delete(x, i)
        y_jack = np.delete(y, i)
        y_err_jack = np.delete(sigma_y, i)
        params, pcov = curve_fit(fun, X_jack, y_jack, sigma = y_err_jack, p0 = guess)
        params_jackknife.append(params)
        pcov_jackknife.append(pcov)
    
    params_mean = np.mean(params_jackknife, axis=0)
    pcov_mean = np.mean(pcov_jackknife, axis=0)
    #print("Parametri del fit non lineare calcolati con il metodo del Jackknife e incertezze su y:")
    #for i in range(len(params_mean)):
    #    print("{}-esimo parametro\n".format(i), params_mean[i], "+/-", np.sqrt(pcov_mean[i][i]))
    return(params_mean, pcov_mean)

def radian_formatter(x, pos):
    # Calcolo della frazione
    # Convertire il numero decimale in una frazione
    if x == 0:
        return '0'
    frazione = Fraction(x).limit_denominator()

    # Ottenere il numeratore e il denominatore
    num = frazione.numerator
    den = frazione.denominator
    if (num == den):
        return '$\pi$'
    elif(den == 1):
        return (f'{x:.0f}$\pi$')
    elif(num == 1):
        return f'$\\frac{{\\pi}}{{{den}}}$'
    else: 
        return f'$\\frac{{{num}}}{{{den}}}\ \pi$'

def x_radiant():
    plt.gca().xaxis.set_major_formatter(FuncFormatter(radian_formatter))

def y_radiant():
    plt.gca().yaxis.set_major_formatter(FuncFormatter(radian_formatter))


def fit_with_bar(func, x, y, guess, sigma=None, maxfev=1000,
                 bounds=(-np.inf, np.inf),
                 ftol=1e-10, xtol=1e-10, gtol=1e-8, diff_step=None,
                 x_scale='jac'):
    """curve_fit con barra di progresso tqdm.notebook.

    Tolleranze di default ragionevoli per dati fisici rumorosi
    (ftol=xtol=1e-10, gtol=1e-8).  Per controllo manuale passarle kw.
    `x_scale='jac'` bilancia parametri con scale molto diverse (default).
    """
    bar_fmt = '{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]'
    with tqdm(total=maxfev, desc=f'Fitting  {func.__name__}',
              unit=' eval', bar_format=bar_fmt) as pbar:
        def _wrap(x, *args):
            pbar.update(1)
            return func(x, *args)
        kw = dict(p0=guess, sigma=sigma, maxfev=maxfev, bounds=bounds,
                  ftol=ftol, xtol=xtol, gtol=gtol, x_scale=x_scale)
        if diff_step is not None:
            kw['diff_step'] = diff_step
        try:
            popt, pcov = curve_fit(_wrap, x, y, **kw)
            pbar.set_description(f'{func.__name__}  OK')
            return popt, pcov
        except RuntimeError as e:
            pbar.set_description(f'{func.__name__}  FAIL')
            raise RuntimeError(f'Fit non convergito: {e}') from None

print('Funzioni definite.')

def plot_observable_4panel(Z_grid, name, label_plain, label_tex,
                           B_grid_1d=None, T_grid_1d=None,
                           name_tex=None,
                           cmap='Viridis',
                           T_slices=(0, 50, 100, 150, 200, 250),
                           B_slices=(1, 5, 20, 40, 60, 80)):
    """Parameters
    ----------
    Z_grid      : array (N_T, N_B) valutato sulla griglia (B_grid_1d, T_grid_1d)
    name        : es. 'ρ_xx'          (titoli plotly -- usa Unicode/testo semplice,
                                       NIENTE LaTeX: plotly rompe il layout)
    label_plain : es. 'ρ_xx'          (hover/colorbar plotly)
    label_tex   : es. r'$\\rho_{xx}$' (ylabel matplotlib)
    name_tex    : es. r'$\\rho_{xx}$' (titoli matplotlib, LaTeX).
                                       Se None, usa `name`.
    B_grid_1d   : array 1D degli assi B (shape Z_grid.shape[1])
    T_grid_1d   : array 1D degli assi T (shape Z_grid.shape[0])
    cmap        : colorscale plotly
    T_slices    : valori di T (K) per i tagli Z(B) a T fissa
    B_slices    : valori di B (T) per i tagli Z(T) a B fissa
    """
    if name_tex is None:
        name_tex = name
    # Fallback: ricostruisci griglie coerenti con Z_grid se non passate
    if B_grid_1d is None:
        B_grid_1d = np.linspace(1.0, 90.0, Z_grid.shape[1])
    if T_grid_1d is None:
        T_grid_1d = np.linspace(0.0, 250.0, Z_grid.shape[0])
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
        title=dict(text=f'{name}(B, T)', font=font1),
        font=font2,                         # default globale: assi, colorbar, sub-titles, tick
        margin=dict(l=10, r=10, t=80, b=10),
    )
    fig_p.show()

    # ---- Matplotlib: Z(B) a T fissa  +  Z(T) a B fissa --------------
    fig, (axB, axT) = plt.subplots(1, 2, figsize=(14, 5))

    col_T = cm.plasma(np.linspace(0.05, 0.85, len(T_slices)))
    for Ts, col in zip(T_slices, col_T):
        i = int(np.argmin(np.abs(T_grid_1d - Ts)))
        axB.plot(B_grid_1d, Z_grid[i, :], color=col, linewidth=2, label=f'T = {Ts} K')
    axB.set_xlabel('B  (T)', fontdict = font2)
    axB.set_ylabel(label_tex, fontdict = font2)
    axB.set_title(f'{name_tex}(B) a T fissa', fontdict = font1)
    axB.legend(fontsize=9); axB.grid(linestyle='-.', alpha=0.5)

    col_B = cm.viridis(np.linspace(0.15, 0.9, len(B_slices)))
    for Bs, col in zip(B_slices, col_B):
        j = int(np.argmin(np.abs(B_grid_1d - Bs)))
        axT.plot(T_grid_1d, Z_grid[:, j], color=col, linewidth=2, label=f'B = {Bs} T')
    axT.set_xlabel('T  (K)', fontdict = font2)
    axT.set_ylabel(label_tex, fontdict = font2)
    axT.set_title(f'{name_tex}(T) a B fissa', fontdict = font1)
    axT.legend(fontsize=9); axT.grid(linestyle='-.', alpha=0.5)

    plt.tight_layout()
    plt.show()