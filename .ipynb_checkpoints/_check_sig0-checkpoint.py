"""Verifica la closed form di sigma_0 nel modello Bessel.

Idea: confrontare sigma_xx_Bessel(B piccolo, ...) con la formula attesa
nel limite B->0. Uso parametri con r < 1 per evitare issue numerici."""
import json, sys, types
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.special import iv

NB = r'C:/Users/vikra/OneDrive/Documenti/File/Uni/Magistrale/Tesi/jupyter tesi/Magnetoresistence.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)
mod = types.ModuleType('nbphys'); mod.np = np; mod.iv = iv
exec(''.join(nb['cells'][3]['source']), mod.__dict__)

# caso con r < 1:  G_is=10, G_an=5 -> G_tilde=10 + 15/8 = 11.875, r=5/11.875=0.421
G_is, G_an, mstar = 10.0, 5.0, 1.0
G_tilde = G_is + 3*G_an/8
r = G_an / G_tilde
print(f'G_is={G_is}, G_an={G_an}, G_tilde={G_tilde:.3f}, r={r:.4f}  (r<1 ? {r<1})')
print(f'closed form 1/sqrt(1-r^2)           = {1/np.sqrt(1-r**2):.6f}')
print(f'closed form pi/sqrt(1-r^2)          = {np.pi/np.sqrt(1-r**2):.6f}')
print(f'closed form pi/(G_tilde*sqrt(1-r^2))= {np.pi/(G_tilde*np.sqrt(1-r**2)):.6f}')

for B in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
    s_xx = mod.sigma_xx_Bessel(np.array([B]), G_is, G_an, mstar)[0]
    alpha = G_an / (8 * mod.K_MEV * B / mstar)
    print(f'  B={B:6.2f} T  alpha={alpha:8.3f}  sigma_xx={s_xx:.6g}')
