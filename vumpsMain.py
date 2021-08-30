import numpy as np
from numpy import linalg as LA
# import scipy.linalg as linalg
# from scipy.sparse.linalg import LinearOperator, eigs
# from scipy.linalg import expm
# from ncon import ncon
from vumpsMPO import vumpsMPO
from tdvpSG2 import tdvpSG
from MPOs import *

def getMixedMPS(d, m):
    C = np.random.rand(m)
    C = C / LA.norm(C)
    AL = (LA.svd(np.random.rand(m * d, m), full_matrices=False)[0]).reshape(m, d, m)
    AR = (LA.svd(np.random.rand(m, m * d), full_matrices=False)[2]).reshape(m, d, m)
    return AL, C, AR

# delta = -0.8 ~ 0.5, m = 0.2
# delta = 0.5, m = 0 ~ 0.5
# M = Thirring(delta = 0.5, m = 0, la = 2) # figure A.7
# M = Thirring(delta = -0.8, m = 0.2, la = 2) # Main : energy = -0.4632171083 ~ -0.4632180828
# M, ML, MR = xy()
# M = np.transpose(M, (1,0,2,3))

# M, ML, MR = ising(h = 3)
# M = np.transpose(M, (1,0,2,3))
# M = Thirring(delta = 2.0, m = 0, la = 2)
# d = 4
# m = 20
# M = ising(h = 10)
# AL, C, AR = getMixedMPS(d, m)
# AL, C, AR, LW, RW, e, cc, ee = vumpsMPO(M, AL, AR, C, m, maxit = 50) # Find ground state with VUMPS
# M = ising(h = 3)
# savename = "isingD%d"%m
# As, Lambdas, datas = tdvpSG(A0 = AL, W = M, dt = 0.01, numiter = 600, backup_step = 1000, bicg_tol = 1e-12, RK4 = True, filename = savename)
# np.save('datas_' + savename, datas)

d = 4
m = 40
M = ising(h = 10)
AL, C, AR = getMixedMPS(d, m)
AL, C, AR, LW, RW, e, cc, ee = vumpsMPO(M, AL, AR, C, m, maxit = 50) # Find ground state with VUMPS
M = ising(h = 3)
savename = "isingD%d"%m
As, Lambdas, datas = tdvpSG(A0 = AL, W = M, dt = 0.01, numiter = 1000, backup_step = 1000, bicg_tol = 1e-12, RK4 = True, filename = savename)
np.save('datas_' + savename, datas)