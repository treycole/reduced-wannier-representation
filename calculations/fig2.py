from wanpy.wpythtb import *
from pythtb import *
from models import Haldane
import numpy as np


# Haldane tight-binding parameters
delta = 1
t = 1
t2 = -0.3 #-0.1

n_super_cell = 2
model = Haldane(delta, t, t2).make_supercell([[n_super_cell, 0], [0, n_super_cell]])
model_name = "haldane"
param_name = f"Delta={delta}_t={t}_t2={t2}"

lat_vecs = model.get_lat()
orb_vecs = model.get_orb()
n_orb = n_orb = model.get_num_orbitals()
n_occ = int(n_orb/2)

low_E_sites = np.arange(0, n_orb, 2)
high_E_sites = np.arange(1, n_orb, 2)

u_wfs_full = wf_array(model, [20, 20])
u_wfs_full.solve_on_grid([0, 0])
chern = u_wfs_full.berry_flux([i for i in range(n_occ)])/(2*np.pi)

model_str = f'C={chern:.1f}_Delta={delta}_t={t}_t2={t2}'

print(f"Low energy sites: {low_E_sites}")
print(f"High energy sites: {high_E_sites}")
print(f"Chern # occupied: {chern: .1f}")
print(model_str)

random = False
low_E = True
omit = False

if random:
    omit_num = 0
    n_tfs = n_occ - omit_num
    tf_list = ["random", n_tfs]
elif omit:
    omit_sites = 6
    tf_list = list(np.setdiff1d(low_E_sites, [omit_sites])) # delta on lower energy sites omitting the last site
    # np.random.choice(low_E_sites, n_tfs, replace=False)
    n_tfs = len(tf_list)
elif low_E:
    tf_list = list(low_E_sites)
    n_tfs = len(tf_list)

Wan_frac = n_tfs/n_occ

if random:
    sv_sfx = model_str + f'_tfxs={tf_list}'
else:
    sv_sfx = model_str + f'_tfx={np.array(tf_list, dtype=int)}'

sv_prefix = f'{model_name}_sing_vals'
sv_dir = 'data'
file_name = f"{sv_dir}/{sv_prefix}_{sv_sfx}"

print(f"# of Wannier functions: {n_tfs}")
print(f"# of occupied bands: {n_occ}")
print(f"Wannier fraction: {Wan_frac}")
print(file_name)

WF = Wannier(model, [20, 20])
twfs = WF.get_trial_wfs(tf_list)
state_idx = list(range(n_occ))

def overlap_mat(psi_wfs, tfs, state_idx):
    """
    Returns A_{k, n, j} = <psi_{n,k} | t_{j}> where psi are Bloch states and t are
    the trial wavefunctions.

    Args:
        psi_wfs (np.array): Bloch eigenstates
        tfs (np.array): trial wfs
        state_idx (list): band indices to form overlap matrix with

    Returns:
        A (np.array): overlap matrix
    """
    psi_wfs = np.take(psi_wfs, state_idx, axis=-2)
    A = np.einsum("...ij, kj -> ...ik", psi_wfs.conj(), tfs)
    return A

# Haldane
k_path = [[0, 0], [2/3, 1/3], [.5, .5], [1/3, 2/3], [0, 0], [.5, .5]]
label = (r'$\Gamma $',r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $', r'$M$')
nk = 201
(k_vec, k_dist, k_node) = model.k_path(k_path, nk, report=False)

eigvals, eigvecs = model.solve_all(k_vec, eig_vectors=True)
eigvecs = np.transpose(eigvecs, axes=(1,0,2))

A = overlap_mat(eigvecs, twfs, state_idx)
V, S, Wh = np.linalg.svd(A, full_matrices=False)

np.save(file_name, S)

nks = 20, 20

u_wfs_full = wf_array(model, [nks[0], nks[1]])
u_wfs_full.solve_on_grid([0, 0])
u_wfs_full = u_wfs_full._wfs

A = overlap_mat(u_wfs_full, twfs, state_idx)
V, S, Wh = np.linalg.svd(A, full_matrices=False)
np.save(file_name+"_full_mesh", S)