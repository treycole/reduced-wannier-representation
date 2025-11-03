import os
import sys

sys.path.insert(0, os.path.abspath("..")) 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from models import Haldane
from pythtb import *
from wanpy import *

# tight-binding parameters
delta = 1
t = 1
t2 = -0.3

n_super_cell = 2
model = Haldane(delta, t, t2).make_supercell([[n_super_cell, 0], [0, n_super_cell]])

#############

n_orb = model.get_num_orbitals()
lat_vecs = model.get_lat()
orb_vecs = model.get_orb()
low_E_sites = np.arange(0, n_orb, 2)
high_E_sites = np.arange(1, n_orb, 2)
n_occ = int(n_orb/2)

u_wfs_full = Bloch(model, 20, 20)
u_wfs_full.solve_model()
chern = Bloch.chern_num()
model_str = f'C={chern:.1f}_Delta={delta}_t={t}_t2={t2}'

print(f"Low energy sites: {low_E_sites}")
print(f"High energy sites: {high_E_sites}")
print(f"Chern # occupied: {chern: .1f}")

### Trial wavefunctions

omit_sites = 6
tf_sites = list(np.setdiff1d(low_E_sites, [omit_sites])) # delta on lower energy sites omitting the last site
tf_list = [ [(orb, 1) ] for orb in tf_sites ]

n_tfs = len(tf_list)
Wan_frac = n_tfs/n_occ

save_sfx = model_str + f'_tfx={np.array(tf_sites, dtype=int)}'

print(f"Trial wavefunctions: {tf_list}")
print(f"# of Wannier functions: {n_tfs}")
print(f"# of occupied bands: {n_occ}")
print(f"Wannier fraction: {Wan_frac}")
print(save_sfx)

############
loc_steps = {}

nks = 20, 20
WF = Wannier(model, nks)
WF.set_trial_wfs(tf_list)

# Single shot projection
WF.single_shot(tf_list)
loc_steps["P"] = {"Omega": WF.spread, "Omega_i": WF.omega_i, "Omega_til": WF.omega_til, "centers": WF.get_centers()}

iter_num = 100000
WF.max_loc(eps=1e-3, iter_num=iter_num, tol=1e-30, grad_min=1e-10, verbose=True)
loc_steps["P+ML"] = {
    "Omega": WF.spread, "Omega_i": WF.omega_i, "Omega_til": WF.omega_til,
    "centers": WF.get_centers(), "iter_num": iter_num}
WF.report()

####################

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

# max-loc
iter_num = 100000
WF.max_loc(eps=1e-3, iter_num=iter_num, tol=1e-10, grad_min=1e-11, verbose=True)

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

