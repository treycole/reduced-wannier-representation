from wanpy.wpythtb import *
from pythtb import *
from models import kane_mele

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


# tight-binding parameters
onsite = 1.0  # 1.0 for top, 2.5 for triv
t = 1
soc = 0.6*t*0.5
rashba = 0.25*t

n_supercell = 3
model = kane_mele(onsite, t, soc, rashba).make_supercell([[n_supercell, 0], [0, n_supercell]])

#############

n_orb = model.get_num_orbitals()
lat_vecs = model.get_lat()
orb_vecs = model.get_orb()
n_occ = int(n_orb/2)*2
low_E_sites = list(np.arange(0, n_orb, 2))
high_E_sites = list(np.arange(1, n_orb, 2))

model_name = "kane_mele"
model_str = f't={t}_soc={soc}_onsite={onsite}_n_occ={n_occ}'
name = f'Wan_frac_{model_name}_{model_str}'
sv_dir = 'data'
if not os.path.exists(sv_dir):
    os.makedirs(sv_dir)

n_tfs = np.array([i for i in range(n_occ-2, 0, -2)])
Wan_frac = n_tfs/n_occ

print(name)

def compute_WFs(n_tf):
    WFs = Wannier(model, [20, 20])

    tf_sites = np.random.choice(low_E_sites, n_tf//2, replace=False) 
    tf_list = [ [(orb, spin, 1) ] for orb in tf_sites for spin in [0,1] ]
    
    WFs.set_trial_wfs(tf_list)
    WFs.single_shot(band_idxs=list(range(n_occ)))
    WFs.subspace_selec(iter_num=50000, tol=1e-3, verbose=True)
    WFs.single_shot(tilde=True)
    WFs.max_loc(eps=2e-3, iter_num=50000, tol=1e-3, grad_min=1e-1, verbose=True)

    return WFs

spread_dict = {}
for idx, n_tf in enumerate(n_tfs):
    print(n_tf, Wan_frac[idx])

    WFs = compute_WFs(n_tf)

    spread_dict[n_tf] = {
            'spread': WFs.spread, 'omega_til': WFs.omega_til, 'omega_i': WFs.omega_i
        }
    np.save(f"{sv_dir}/{name}_spread_dict.npy", spread_dict)

    if n_tf == n_tfs[0]:
        np.save(f"{sv_dir}/{name}_WF_16o18.npy", WFs)
