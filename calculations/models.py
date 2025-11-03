import numpy as np
from wanpy import Model

# used for testing purposes

def Haldane(delta, t, t2):
    lat = [[1, 0], [0.5, np.sqrt(3)/2]]
    orb = [[1/3, 1/3], [2/3, 2/3]]

    model = Model(2, 2, lat, orb)

    model.set_onsite([-delta, delta], mode='reset')

    for lvec in ([0, 0], [-1, 0], [0, -1]):
        model.set_hop(t, 0, 1, lvec, mode='reset')
        model.set_hop(t, 0, 1, lvec, mode='reset')

    for lvec in ([1, 0], [-1, 1], [0, -1]):
        model.set_hop(t2*1j, 0, 0, lvec, mode='reset')
        model.set_hop(t2*-1j, 1, 1, lvec, mode='reset')

    return model


def kane_mele(onsite, t, soc, rashba):
  "Return a Kane-Mele model in the normal or topological phase."

  # define lattice vectors
  lat = [[1, 0], [0.5, np.sqrt(3)/2]]
  # define coordinates of orbitals
  orb = [[1/3, 1/3], [2/3, 2/3]]
 
  # make two dimensional tight-binding Kane-Mele model
  ret_model = Model(2, 2, lat, orb, nspin=2)

  # set on-site energies
  ret_model.set_onsite([-onsite, onsite])

  # useful definitions
  sigma_x = np.array([0.,1.,0.,0])
  sigma_y = np.array([0.,0.,1.,0])
  sigma_z = np.array([0.,0.,0.,1])

  # set hoppings (one for each connected pair of orbitals)
  # (amplitude, i, j, [lattice vector to cell containing j])
  # spin-independent first-neighbor hoppings
  ret_model.set_hop(t, 0, 1, [ 0, 0])
  ret_model.set_hop(t, 0, 1, [ 0,-1])
  ret_model.set_hop(t, 0, 1, [-1, 0])

  # second-neighbour spin-orbit hoppings (s_z)
  ret_model.set_hop(-1.j*soc*sigma_z, 0, 0, [ 0, 1])
  ret_model.set_hop( 1.j*soc*sigma_z, 0, 0, [ 1, 0])
  ret_model.set_hop(-1.j*soc*sigma_z, 0, 0, [ 1,-1])
  ret_model.set_hop( 1.j*soc*sigma_z, 1, 1, [ 0, 1])
  ret_model.set_hop(-1.j*soc*sigma_z, 1, 1, [ 1, 0])
  ret_model.set_hop( 1.j*soc*sigma_z, 1, 1, [ 1,-1])

  # Rashba first-neighbor hoppings: (s_x)(dy)-(s_y)(d_x)
  r3h = np.sqrt(3.0)/2.0
  # bond unit vectors are (r3h,half) then (0,-1) then (-r3h,half)
  ret_model.set_hop(1.j*rashba*( 0.5*sigma_x-r3h*sigma_y), 0, 1, [ 0, 0], mode="add")
  ret_model.set_hop(1.j*rashba*(-1.0*sigma_x            ), 0, 1, [ 0,-1], mode="add")
  ret_model.set_hop(1.j*rashba*( 0.5*sigma_x+r3h*sigma_y), 0, 1, [-1, 0], mode="add")

  return ret_model





