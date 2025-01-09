from numpy import sqrt
from wanpy.wpythtb import Model

# used for testing purposes

def Haldane(delta, t, t2):
    lat = [[1, 0], [0.5, sqrt(3)/2]]
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