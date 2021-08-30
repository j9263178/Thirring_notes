import numpy as np
from numpy import linalg
import time
from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigsh, eigs, gmres, bicgstab
from scipy.linalg import polar
from typing import List, Optional

#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='

def vumpsMPO(W, AL, AR, C, m, eigsolve_tol = 1e-5, grad_tol = 1e-8, energy_tol = 1e-8, maxit = 10, update_mode = "polar"):

    def getR(AL):
        TL = ncon([AL, AL.conj()], [[-1,1,-3], [-2,1,-4]]).reshape(m**2, m**2)
        # TL_ = np.transpose(TL, (0, 2, 1, 3))
        R = eigs(TL, k = 1, which = 'LM')[1].reshape(m, m)
        R = 0.5*(R+R.conj().T)
        R/=ncon([R,np.eye(m)],[[1,2],[1,2]])
        return np.real(R)

    def getL(AR):
        TR = ncon([AR, AR.conj()], [[-3,1,-1], [-4,1,-2]]).reshape(m**2, m**2)
        # TR_ = np.transpose(TR, (0, 2, 1, 3))
        # TR = TR.T
        L = eigs(TR, k = 1, which = 'LM')[1].reshape(m, m)
        L = 0.5*(L+L.conj().T)
        L /= ncon([L,np.eye(m)],[[1,2],[1,2]])
        return np.real(L)

    def getLWaC21(LWa, YLa, TL, la):
        m = LWa.shape[0]
        def Lcontract(LWa):
            m = TL.shape[0]
            b = TL.shape[1]
            LWa_TL = ncon([LWa, TL],[[1,2],[1,-1,2,-2]])
            return LWa.flatten() - la*LWa_TL.flatten()

        LeftOp = LinearOperator((m**2, m**2), matvec=Lcontract, dtype=np.float64)
        LWa_temp, is_conv = gmres(LeftOp, YLa.flatten(), x0=LWa.flatten(), tol=1e-10,
                                restart=None, atol=None)
        LWa_temp = LWa_temp.reshape(m, m)
        # La = 0.5 * (La_temp + La_temp.T) investgate
        return LWa_temp
    
    def getRWaC21(RWa, YRa, TR, la):
        m = RWa.shape[0]
        def Rcontract(RWa):
            m = TR.shape[0]
            b = TR.shape[1]
            RWa_TR = ncon([RWa, TR],[[1,2],[-1,1,-2,2]])
            return RWa.flatten() - la*RWa_TR.flatten()

        RightOp = LinearOperator((m**2, m**2), matvec=Rcontract, dtype=np.float64)
        RWa_temp, is_conv = gmres(RightOp, YRa.flatten(), x0=RWa.flatten(), tol=1e-10,
                                restart=None, atol=None)
        RWa_temp = RWa_temp.reshape(m, m)
        # La = 0.5 * (La_temp + La_temp.T) investgate
        return RWa_temp

    def getLWaC25(LWa, YLa, AL, R):

        def Lcontract(v):
            m = AL.shape[0]
            v = v.reshape(m,m)
            # LWa_TL = ncon([v, TL],[[1,2],[1,-1,2,-2]])
            LWa_TL = ncon([v, AL, AL.conj()],[[1,2],[1,3,-1],[2,3,-2]])
            LWa_P = ncon([v, R], [[1,2],[1,2]])*np.eye(m)
            return v.flatten() - LWa_TL.flatten() + LWa_P.flatten()

        LeftOp = LinearOperator((m**2, m**2), matvec=Lcontract, dtype=np.float64)

        B = YLa - ncon([YLa, R], [[1,2],[1,2]])*np.eye(m)

        LWa_temp, is_conv = bicgstab(LeftOp, B.flatten(), x0=LWa.flatten(), tol=1e-12)

        LWa_temp = np.real(LWa_temp).reshape(m, m)

        return LWa_temp

    def getRWaC25(RWa, YRa, AR, L):

        def Rcontract(v):
            m = AR.shape[0]
            v = v.reshape(m,m)
            # RWa_TR = ncon([TR, v],[[-1,1,-2,2], [1,2]])
            RWa_TR = ncon([v, AR, AR.conj()],[[1,2], [-1,3,1], [-2,3,2]])
            RWa_P = ncon([L, v], [[1,2],[1,2]])*np.eye(m)
            return v.flatten() - RWa_TR.flatten() + RWa_P.flatten()

        RightOp = LinearOperator((m**2, m**2), matvec=Rcontract, dtype=np.float64)

        B = YRa - ncon([L, YRa], [[1,2],[1,2]])*np.eye(m)

        RWa_temp, is_conv = bicgstab(RightOp, B.flatten(), x0=RWa.flatten(), tol=1e-12)

        RWa_temp = np.real(RWa_temp).reshape(m, m)

        return RWa_temp

    def getLW(LW, AL):
        m = AL.shape[0]
        dw = W.shape[0]
        YL = [np.zeros([m,m]) for i in range(dw)]
        LW_ = LW.copy()
        LW_[dw-1] = np.eye(m)
        R = getR(AL)
        for a in range(dw-2, -1, -1):
            for b in range(a+1, dw):
                YL[a] += ncon([LW_[b], AL, W[b,a], AL.conj()],[[1,2], [1,4,-1], [4,5], [2,5,-2]])
            if W[a, a, 0, 0] == 0:
                LW_[a] = YL[a]
            elif W[a, a, 0, 0] == 1:
                LW_[a] = getLWaC25(LW_[a], YL[a], AL, R)
            else:  #W[a,a] = constant*I
                LW_[a] = getLWaC21(LW_[a], YL[a], AL, W[a, a, 0, 0])
        
        return np.asarray(LW_), ncon([YL[0], R], [[1,2],[1,2]])

    def getRW(RW, AR):
        m = AR.shape[0]
        dw = W.shape[0]
        YR = [np.zeros([m,m]) for i in range(dw)]
        RW_ = RW.copy()
        RW_[0] = np.eye(m)
        L = getL(AR)

        for a in range(1, dw):
            for b in range(a-1, -1, -1):
                YR[a] += ncon([RW_[b], AR, W[a,b], AR.conj()],[[1,2], [-1,4,1], [4,5], [-2,5,2]])
            if W[a, a, 0, 0] == 0:
                RW_[a] = YR[a]
            elif W[a, a, 0, 0] == 1:
                RW_[a] = getRWaC25(RW_[a], YR[a], AR, L)
            else:  #W[a,a] = a constant*I
                RW_[a] = getRWaC21(RW_[a], YR[a], AR, W[a, a, 0, 0])

        return np.asarray(RW_), ncon([L, YR[-1]], [[1,2],[1,2]])

    def applyH_AC(AC, HAC):
        LW, W, RW = HAC
        d = AC.shape[1]
        m = AC.shape[0]

        def MidTensor(AC_mat):
            AC_mat=AC_mat.reshape([m, d, m])
            tensors = [LW, W, RW, AC_mat]
            labels = [[2,1,-1], [2,3,4,-2], [3,5,-3], [1,4,5]]
            return (ncon(tensors, labels)).flatten()

        TensorOp = LinearOperator((d * m**2, d * m**2),
                                matvec=MidTensor, dtype=np.float64)

        AC_new = eigsh(TensorOp, k=1, which='SA', v0=AC.flatten(),
                    ncv=None, maxiter=None, tol=eigsolve_tol)[1].reshape(m, d, m)
        return AC_new

    def applyH_C(C, HC):
        LW, RW = HC
        def MidWeights(C_mat):
            C_mat = C_mat.reshape(m, m)
            tensors = [LW, C_mat, RW]
            labels = [[2,1,-1], [1,3], [2,3,-2]]
            con_order = [1,3,2]
            return (ncon(tensors, labels)).flatten()
        WeightOp = LinearOperator((m**2, m**2), matvec=MidWeights, dtype=np.float64)

        C_temp = eigsh(WeightOp, k=1, which='SA', v0=np.diag(C).flatten(),
                    ncv=None, maxiter=None, tol=eigsolve_tol)[1]
        ut, C_new, vt = linalg.svd(C_temp.reshape(m, m)) # norm ~= 1.

        return ut, C_new, vt

    def updateALAR(AL, AR, AC, C):
        if update_mode == 'polar':
            AL = (polar(AC.reshape(m * d, m))[0]).reshape(m, d, m)
            AR = (polar(AC.reshape(m, d * m), side='left')[0]
                    ).reshape(m, d, m)
        elif update_mode == 'svd':
            ut, _, vt = linalg.svd(AC.reshape(m * d, m) @ np.diag(C), full_matrices=False)
            AL = (ut @ vt).reshape(m, d, m)
            ut, _, vt = linalg.svd(np.diag(C) @ AC.reshape(m, d * m), full_matrices=False)
            AR = (ut @ vt).reshape(m, d, m)
        return AL, AR

    def getEE(C):
        ee = 0
        for x in C:
            ee += -(x**2 * np.log2(x**2))
        # print("EE = %.10f"%ee)
        return ee

    def getCC(AC):
        sZ =  1/2 * np.array([[1.0, 0], [0, -1.0]])
        sI =  np.array([[1.0, 0], [0, 1.0]])
        meaa, meab = np.kron(sI,sZ), np.kron(sZ,sI)
        a = ncon([AC, meaa, AC.conj()], [[1,2,3],[2,4],[1,4,3]])
        b = ncon([AC, meab, AC.conj()], [[1,2,3],[2,4],[1,4,3]])
        cc = 0.5*(a-b)
        # print("CC = %.10f"%cc)
        return cc

    d = AL.shape[1] # physical dimension
    dw = W.shape[0]

    # m is the bond dimension 
    if m is None:
        m = AL.shape[0]
    else:
        if m > AL.shape[0]:
        # Expand tensors to new dimension
            AL = Orthogonalize(TensorExpand(AL, [m, d, m]), 2)
            C = TensorExpand(C, [m])
            AR = Orthogonalize(TensorExpand(AR, [m, d, m]), 2)

    AC = ncon([AL, np.diag(C)], [[-1, -2, 1], [1, -3]])
    LW = np.asarray([np.random.rand(m,m) for i in range(dw)])
    RW = np.asarray([np.random.rand(m,m) for i in range(dw)])
    el, er = 9999, 9999
    Energy, Energynew, mindiff = 0, 0, 9999.0
    # main algorithm here
    # while(True):
    for k in range(maxit):

        # time_start = time.time()

        LW, EnergyL = getLW(LW, AL)
        RW, EnergyR = getRW(RW, AR)

        Energynew = (EnergyL+EnergyR)/4
        print("iteration %d, energy = %.10f"%(k,Energynew))

        Ediff = np.abs(Energynew-Energy)
        Energy = Energynew
        mindiff = min(Ediff, mindiff)

        if np.abs(max(el,er))<grad_tol or Ediff<energy_tol:
            cc = getCC(AC)
            print("Tolerence achieved!")
            break

        if Ediff<1e-4:
            eigsolve_tol = 1e-8
        if Ediff<1e-6:
            eigsolve_tol = 1e-10

        ut, C, vt = applyH_C(C, (LW, RW))
        AL = ncon([ut.T, AL, ut], [[-1, 1], [1, -2, 2], [2, -3]])
        AR = ncon([vt, AR, vt.T], [[-1, 1], [1, -2, 2], [2, -3]])

        LW, _ = getLW(LW, AL)
        RW, _ = getRW(RW, AR)
        AC = applyH_AC(AC, (LW, W, RW))

        AL, AR = updateALAR(AL, AR, AC, C)

        ALC = ncon([AL, np.diag(C)],[[-1,-2,1],[1,-3]])
        ARC = ncon([np.diag(C),AR],[[-1,1],[1,-2,-3]])
        el = linalg.norm(ALC-AC) 
        er = linalg.norm(ARC-AC)

        # time_end = time.time()
        # print("total time: "+str(time_end - time_start))
        # ee = getEE(C)
        # print(ee)

    print("Energy minimum diff : %.15f"%mindiff)
    return AL, C, AR, LW, RW, Energy, cc, _


def TensorExpand(A: np.ndarray, chivec: List):
  """ expand tensor dimension by padding with zeros """

  if [*A.shape] == chivec:
    return A
  else:
    for k in range(len(chivec)):
      if A.shape[k] != chivec[k]:
        indloc = list(range(-1, -len(chivec) - 1, -1))
        indloc[k] = 1
        A = ncon([A, np.eye(A.shape[k], chivec[k])], [indloc, [1, -k - 1]])

    return A

def Orthogonalize(A: np.ndarray, pivot: int):
  """ orthogonalize an array with respect to a pivot """

  A_sh = A.shape
  ut, st, vht = np.linalg.svd(
      A.reshape(np.prod(A_sh[:pivot]), np.prod(A_sh[pivot:])),
      full_matrices=False)
  return (ut @ vht).reshape(A_sh)




