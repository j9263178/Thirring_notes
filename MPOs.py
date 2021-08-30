import numpy as np
from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigs

def getLR(A):
    m = A.shape[0]
    TM = ncon([A, A.conj()], [[-1,1,-3], [-2,1,-4]]).reshape(m**2, m**2)
    R = eigs(TM, k = 1, which = 'LM')[1].reshape(m, m)
    L = eigs(TM.T, k = 1, which = 'LM')[1].reshape(m, m)
    return L, R

def MPS(Nsites, d, chi):
    A = [0 for x in range(Nsites)]
    A[0] = np.random.rand(1, d,min(chi,d))
    for k in range(1,Nsites):
        A[k] = np.random.rand(A[k-1].shape[2],d,min(min(chi,A[k-1].shape[2]*d),d**(Nsites-k-1)))
    return A

def MixedMPS(d, m):
    C = np.random.rand(m)
    C = C / np.linalg.norm(C)
    AL = (np.linalg.svd(np.random.rand(m * d, m), full_matrices=False)[0]).reshape(m, d, m)
    AR = (np.linalg
    .svd(np.random.rand(m, m * d), full_matrices=False)[2]).reshape(m, d, m)
    return AL, C, AR

def measure1siteoperator(A, O):

    '''
    Assumes A starts in left canonical form (i.e. -U-U-U-...-U-)
    '''
    N = len(A)
    rho = np.eye(A[-1].shape[2])
    expval = np.zeros(N)
    for k in range(N-1, -1, -1):
        A[k] = np.real(A[k])
        expval[k] = ncon([rho, A[k], A[k].conj(), O],[[1,2],[5,3,1],[5,4,2],[3,4]])
        rho = ncon([rho, A[k], A[k].conj()],[[1,2],[-1,3,1],[-2,3,2]])
    return expval

def xxz_xStaggered(d = 2, J = 1, Delta = 2, h = 0, H = 0):

    d = 2

    sx = 1/2*np.array([[0, 1], [1, 0]])
    sy = 1/2*np.array([[0, -1j], [1j, 0]])
    sz = 1/2*np.array([[1, 0], [0,-1]])

    sp = sx+1j*sy
    sm = sx-1j*sy
    eye = np.eye(2)

    M = np.zeros([5, 5, d, d])
    M[0,0] = M[4,4] = eye
    M[0,1] = sp.real()
    M[0,2] = sm.real()
    M[0,3] = sz.real()
    M[1,4] = J/2 * sm.real()
    M[2,4] = J/2 * sp.real()
    M[3,4] = Delta * J * sz.real()
    M[0,4] = - h*sx.real() + H*sx.real()

    M_ = np.zeros([5, 5, d, d])
    M_[0,0] = M_[4,4] = eye
    M_[0,1] = sp.real()
    M_[0,2] = sm.real()
    M_[0,3] = sz.real()
    M_[1,4] = J/2 * sm.real()
    M_[2,4] = J/2 * sp.real()
    M_[3,4] = Delta * J * sz.real()
    M_[0,4] = - h*sx.real() - H*sx.real()

    M = cy.UniTensor(M,0)
    M_ = cy.UniTensor(M_,0)
    L0 = cy.zeros([5,d,d])
    R0 = cy.zeros([5,d,d])
    L0 = cy.UniTensor(cy.zeros([5,1,1]),0)
    R0 = cy.UniTensor(cy.zeros([5,1,1]),0)
    L0.get_block_()[0,0,0] = 1.; R0.get_block_()[4,0,0] = 1.
    return M, M_, L0, R0

def xxz_zStaggered(J = 1, h = 0, Delta = 0, Nsites = 100):

    d = 2

    sx = 1/2*np.array([[0, 1], [1, 0]])
    sy = 1/2*np.array([[0, -1j], [1j, 0]])
    sz = 1/2*np.array([[1, 0], [0,-1]])

    sp = np.real(sx+1j*sy)
    sm = np.real(sx-1j*sy)
    eye = np.eye(2)

    M = np.zeros([5, 5, d, d])
    M[0,0] = M[4,4] = eye
    M[0,1] = sp; M[1,4] = J/2 * sm
    M[0,2] = sm; M[2,4] = J/2 * sp
    M[0,3] = sz; M[3,4] = Delta * J * sz
    M[0,4] = - J*h*sz

    M_ = np.copy(M)
    M_[0,4] = + J*h*sz

    ML = np.array([1,0,0,0,0]).reshape(5,1,1)
    MR = np.array([0,0,0,0,1]).reshape(5,1,1)

    Ms = []
    for i in range(Nsites):
        if i%2:
            Ms.append(M)
        else:
            Ms.append(M_)    

    return Ms, ML, MR

def xy(gamma = 0.3, g = 0.2):

    d=2

    sx = 1/2*np.array([[0, 1], [1, 0]])
    sy = 1/2*np.array([[0, -1j], [1j, 0]])
    sz = 1/2*np.array([[1, 0], [0,-1]])

    sp = sx+1j*sy
    sm = sx-1j*sy
    eye = np.eye(2)
    
    M = np.zeros([4, 4, d, d])
    M[0,0] = M[3,3] = eye
    M[0,1] = -sp
    M[0,2] = -sm
    M[1,3] = 1/2 * sm + gamma/2 *sp
    M[2,3] = 1/2 * sp + gamma/2 *sm
    M[0,3] = -g*sz

    ML = np.array([1,0,0,0]).reshape(4,1,1)
    MR = np.array([0,0,0,1]).reshape(4,1,1)
    return M, ML, MR

def xx(mu = 0):
    
    d = 2
    sx = 1/2*np.array([[0, 1], [1, 0]])
    sy = 1/2*np.array([[0, -1j], [1j, 0]])
    sz = 1/2*np.array([[1, 0], [0,-1]])
    sP = np.real(sx+1j*sy)
    sM = np.real(sx-1j*sy)
    sI = np.array([[1, 0], [0, 1]])
    M = np.zeros([4, 4, d, d])
    M[0,0,:,:] = sI; M[3,3,:,:] = sI
    M[0,1,:,:] = sM; M[1,3,:,:] = 1/2 * sP
    M[0,2,:,:] = sP; M[2,3,:,:] = 1/2 * sM
    M[0,3,:,:] = mu * sz
    ML = np.array([1,0,0,0]).reshape(4,1,1)
    MR = np.array([0,0,0,1]).reshape(4,1,1)
    h = 1/2 * (np.kron(sP,sM)+np.kron(sM,sP)).reshape(d,d,d,d)
    return M, ML, MR, h

def xxz_zh(Delta = -0.5, h = 1):

    J = 1
    d = 2

    sx = 1/2*np.array([[0, 1], [1, 0]])
    sy = 1/2*np.array([[0, -1j], [1j, 0]])
    sz = 1/2*np.array([[1, 0], [0,-1]])

    # sx = np.array([[0, 1], [1, 0]])
    # sy = np.array([[0, -1j], [1j, 0]])
    # sz = np.array([[1, 0], [0,-1]])

    sp = sx+1j*sy
    sm = sx-1j*sy
    eye = np.eye(2)

    M = np.zeros([5, 5, d, d])
    M[0,0] = eye; M[4,4] = eye
    M[0,1] = sp;  M[1,4] = -J/2 * sm
    M[0,2] = sm;  M[2,4] = -J/2 * sp
    M[0,3] = sz;  M[3,4] = -Delta * J * sz
    M[0,4] = -h*sz

    ML = np.array([1,0,0,0,0]).reshape(5,1,1)
    MR = np.array([0,0,0,0,1]).reshape(5,1,1)
    return M, ML, MR

def ising(h = 0):
    d = 2
    dw = 3
    sx = np.array([[0, 1], [1, 0]]) #Pauli!
    sz = np.array([[1, 0], [0,-1]]) #Pauli!
    sI = np.array([[1, 0], [0, 1]])
    M = np.zeros([3, 3, d, d])
    M[0,0,:,:] = sI; M[2,2,:,:] = sI
    M[0,1,:,:] = sx; M[1,2,:,:] = sx
    M[0,2,:,:] = h * sz
    M = np.transpose(M, (1,0,2,3))
    # ML = np.array([1,0,0]).reshape(3,1,1)
    # MR = np.array([0,0,1]).reshape(3,1,1)
    d = 4
    M = ncon([M, M],[[-1,1,-3,-5],[1,-2,-4,-6]]).reshape(dw, dw, d, d)
    return M

def Thirring(delta = 2.0, m = 0.0, la = 3):
    dw = 6
    d = 2
    a = 1
    sX = 1/2* np.array([[0, 1], [1, 0]])
    sY = 1/2* np.array([[0, -1j], [1j, 0]])
    sP = np.real(sX+1j*sY)
    sM = np.real(sX-1j*sY)
    sZ = 1/2 * np.array([[1, 0], [0,-1]])
    sI = np.array([[1, 0], [0, 1]])
    M1 = np.zeros([dw, dw, d, d])
    M1[0,0,:,:] = sI; M1[5,1,:,:] = -1/(2*a) * sP
    M1[1,0,:,:] = sM; M1[5,2,:,:] = -1/(2*a) * sM
    M1[2,0,:,:] = sP; M1[5,3,:,:] = 2*la * sZ
    M1[3,0,:,:] = sZ; M1[5,4,:,:] = delta/a * sZ
    M1[4,0,:,:] = sZ; M1[5,5,:,:] = sI
    M1[5,0,:,:] = (delta/a + m) * sZ + (la/4 + delta/(4*a)) * sI
    M1[3,3,:,:] = sI

    M2 = np.copy(M1)
    M2[5,0,:,:] = (delta/a - m) * sZ + (la/4 + delta/(4*a)) * sI

    d = 4
    M = ncon([M1, M2],[[-1,1,-3,-5],[1,-2,-4,-6]]).reshape(dw, dw, d, d)
    return M

def FermionCr(AL, filename):

    m = AL.shape[0] # bond dim
    sX = 1/2 * np.array([[0, 1], [1, 0]])
    sY = 1/2 * np.array([[0, -1j], [1j, 0]])
    sP = np.real(sX+1j*sY)
    sM = np.real(sX-1j*sY)
    sZ = 1/2 * np.array([[1, 0], [0,-1]])

    damp = expm(1j*np.pi*sZ) # scipy.linalg.expm
    O1 = np.kron(sP, damp)
    O2 = np.kron(damp, damp)
    O3 = np.kron(damp, sM)

    # get the left and right eigenvector of transfer matrix
    TL = ncon([AL, AL.conj()], [[-1,1,-3], [-2,1,-4]]).reshape(m**2, m**2)
    R = eigs(TL, k = 1, which = 'LM')[1].reshape(m, m)
    R /= ncon([R,np.eye(m)],[[1,2],[1,2]])
    L = np.eye(m)

    AO1A = ncon([AL, O1, AL.conj()], [[-1,1,-3], [1,2], [-2,2,-4]])
    AO3A = ncon([AL, O3, AL.conj()], [[-1,1,-3], [1,2], [-2,2,-4]])
    AO2A = ncon([AL, O2, AL.conj()], [[-1,1,-3], [1,2], [-2,2,-4]])

    Left = ncon([L, AO1A],[[1,2],[1,2,-1,-2]])
    Right = ncon([AO3A, R],[[-1,-2,1,2], [1,2]])

    cur = ncon([Left, AO2A], [[1,2],[1,2,-1,-2]])

    G = np.zeros(200, dtype = np.csingle)
    G[0] = ncon([Left, Right],[[1,2],[1,2]])

    for r in range(1, 200):
        print(r)
        G[r] = ncon([cur, Right], [[1,2],[1,2]])
        cur = ncon([cur, AO2A], [[1,2],[1,2,-1,-2]])

    # np.save("Thrring_mass_Gr1", G)
    np.save(filename, G)


# deltas = np.linspace(-1.0, 2.0, 13)
# ms = np.linspace(0, 0.5, 11)
# E = np.zeros((13,11))
# EE = np.zeros((13,11))
# CC = np.zeros((13,11))
# for i in range(13):
#     for j in range(11):
#         print("Doing delta = %.3f, m = %.3f "%(deltas[i], ms[j]))
#         M = Thirring(delta = deltas[i], m = ms[j])
#         C = np.random.rand(m)
#         C = C / LA.norm(C)
#         AL = (LA.svd(np.random.rand(m * d, m), full_matrices=False)[0]).reshape(m, d, m)
#         AR = (LA.svd(np.random.rand(m, m * d), full_matrices=False)[2]).reshape(m, d, m)
#         AL, C, AR, LW, RW, e, ee, cc = vumpsMPO(M, AL, AR, C, m, tol = 1e-4)
#         E[i,j] = e
#         EE[i,j] = ee
#         CC[i,j] = cc

# d = 4

# ms = [20, 30, 50, 60]
# # m = 32
# # deltas = [-0.1, -0.5, -0.7, -0.9]
# for m in ms:
#     print("doing : %d"%m)
#     M = Thirring(delta = -0.9, m = 0, la = 3)
#     C = np.random.rand(m)
#     C = C / LA.norm(C)
#     AL = (LA.svd(np.random.rand(m * d, m), full_matrices=False)[0]).reshape(m, d, m)
#     AR = (LA.svd(np.random.rand(m, m * d), full_matrices=False)[2]).reshape(m, d, m)
#     AL, C, AR, LW, RW, e, ee, cc = vumpsMPO(M, AL, AR, C, m, tol = 1e-4)
#     FermionCr(AL, "KT_D_"+str(m))
# d = 4
# m = 32
# deltas = [-0.1, -0.5, -0.7, -0.9]
# for delta in deltas:
#     print("doing : %.3f"%delta)
#     M = Thirring(delta = delta, m = 0, la = 3)
#     C = np.random.rand(m)
#     C = C / LA.norm(C)
#     AL = (LA.svd(np.random.rand(m * d, m), full_matrices=False)[0]).reshape(m, d, m)
#     AR = (LA.svd(np.random.rand(m, m * d), full_matrices=False)[2]).reshape(m, d, m)
#     AL, C, AR, LW, RW, e, ee, cc = vumpsMPO(M, AL, AR, C, m, tol = 1e-6)
#     FermionCr(AL, "test_"+str(delta))

# E[i,j] = e
# EE[i,j] = ee
# CC[i,j] = cc
# np.save('Thrring_E', E)
# np.save('Thrring_EE', EE)
# np.save('Thrring_CC', CC)