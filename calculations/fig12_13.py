import os
import sys

sys.path.insert(0, os.path.abspath("..")) 

import numpy as np
from models import kane_mele
from pythtb import *
from wanpy import *


# tight-binding parameters
onsite = 1.0  # 1.0 for top, 2.5 for triv
t = 1
soc = 0.6*t*0.5
rashba = 0.25*t

n_supercell = 2
model = kane_mele(onsite, t, soc, rashba).make_supercell([[n_supercell, 0], [0, n_supercell]])

#############

n_orb = model.get_num_orbitals()
lat_vecs = model.get_lat()
orb_vecs = model.get_orb()
n_occ = int(n_orb/2)*2
low_E_sites = list(np.arange(0, n_orb, 2))
high_E_sites = list(np.arange(1, n_orb, 2))

print(f"Low energy sites: {low_E_sites}")
print(f"High energy sites: {high_E_sites}")

##### Trial wavefunctions #####

omit_sites = 6
tf_sites  = list(np.setdiff1d(low_E_sites, [omit_sites]))  # delta on lower energy sites omitting the last site
tf_list = [ [(orb, spin, 1) ] for orb in tf_sites for spin in [0,1] ]
n_tfs = len(tf_list)
Wan_frac = n_tfs/n_occ

model_name = "kane_mele"
model_str = f't={t}_soc={soc}_onsite={onsite}_n_occ={n_occ}'
save_sfx = model_str + f'_tfx={np.array(tf_sites, dtype=int)}'
print(save_sfx)

print(f"Trial wavefunctions: {tf_list}")
print(f"# of Wannier functions: {n_tfs}")
print(f"# of occupied bands: {n_occ}")
print(f"Wannier fraction: {Wan_frac}")

# ##################
loc_steps = {}

nks = 20, 20
WF = Wannier(model, nks)
WF.set_trial_wfs(tf_list)

# Single shot projection
WF.single_shot(band_idxs=list(range(n_occ)))
loc_steps["P"] = {"Omega": WF.spread, "Omega_i": WF.omega_i, "Omega_til": WF.omega_til, "centers": WF.get_centers()}

# Maximal localization
iter_num = 100000
WF.max_loc(eps=1e-3, iter_num=iter_num, tol=1e-10, grad_min=1e-2, verbose=True)
loc_steps["P+ML"] = {
    "Omega": WF.spread, "Omega_i": WF.omega_i, "Omega_til": WF.omega_til,
    "centers": WF.get_centers(), "iter_num": iter_num}
WF.report()

# ###################
nks = 20, 20
WF = Wannier(model, nks)
WF.set_trial_wfs(tf_list)

# initial projection
WF.single_shot(band_idxs=list(range(n_occ)))

# subspace selection
iter_num = 100000
WF.subspace_selec(iter_num=iter_num, tol=1e-10, verbose=True)

# second projection
WF.single_shot(tilde=True)
loc_steps["P+SS+P"] = {
    "Omega": WF.spread, "Omega_i": WF.omega_i, "Omega_til": WF.omega_til,
    "centers": WF.get_centers(), "iter_num": iter_num}

# maximal localization
iter_num = 100000
WF.max_loc(eps=6e-3, iter_num=iter_num, tol=1e-10, grad_min=1e-2, verbose=True)

loc_steps["P+SS+P+ML"] = {
    "Omega": WF.spread, "Omega_i": WF.omega_i, "Omega_til": WF.omega_til,
    "centers": WF.get_centers(), "iter_num": iter_num}

sv_dir = 'data'
sv_prefix = 'WF_loc_steps'
file_name = f"{sv_dir}/{sv_prefix}_{save_sfx}"
np.save(file_name, loc_steps)

sv_prefix = 'WF_max_loc'
file_name = f"{sv_dir}/{sv_prefix}_{save_sfx}"
np.save(file_name, WF)
