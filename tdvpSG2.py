import numpy as np
import scipy.linalg as linalg
from scipy.sparse.linalg import LinearOperator, eigsh, eigs, gmres, bicgstab
from ncon import ncon

#                                                    __----~~~~~~~~~~~------___
#                                   .  .   ~~//====......          __--~ ~~
#                   -.            \_|//     |||\\  ~~~~~~::::... /~
#                ___-==_       _-~o~  \/    |||  \\            _/~~-
#        __---~~~.==~||\=_    -_--~/_-~|-   |\\   \\        _/~
#    _-~~     .=~    |  \\-_    '-~7  /-   /  ||    \      /
#  .~       .~       |   \\ -_    /  /-   /   ||      \   /
# /  ____  /         |     \\ ~-_/  /|- _/   .||       \ /
# |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\
#          '         ~-|      /|    |-~\~~       __--~~
#                      |-~~-_/ |    |   ~\_   _-~            /\
#                           /  \     \__   \/~                \__
#                       _--~ _/ | .-~~____--~-/                  ~~==.
#                      ((->/~   '.|||' -_|    ~~-/ ,              . _||
#                                 -_     ~\      ~~---l__i__i__i--~~_/
#                                 _-~-__   ~)  \--______________--~~
#                               //.-~~~-~_--~- |-------~~~~~~~~
#                                      //.-~~~--\


def truncate(v):
    D,U = np.linalg.eigh(v)
    for i in range(len(D)):
        if D[i]<0:
            D[i] *= -1 
        if D[i]<1e-15:
            D[i] = np.random.rand(1)[0] * 1e-15
    v_ = U @ np.diag(D) @ U.conj().T
    return v_

def find_lr(TM, m):
    e, v = eigs(TM, k = 1, which = 'LM')
    v = v.reshape(m, m)
    v = 0.5*(v+v.conj().T)
    # print(min(np.linalg.norm(v_-v), np.linalg.norm(v_+v)))
    return truncate(v)

def toLambdaGamma(AL):
    chil, d, chir = AL.shape[0], AL.shape[1], AL.shape[2]
    u, s, vt = linalg.svd(AL.reshape(chil*d,chir), full_matrices=False)
    Gamma = ncon([vt, u.reshape(chil, d, chir)],[[-1,1],[1,-2,-3]])
    Lambda = np.diag(s)/linalg.norm(s)
    return Lambda, Gamma

def toCanonical(Lambda, Gamma):
    m = Lambda.shape[0]
    Ltemp = ncon([Lambda, Gamma, Lambda, Gamma.conj()], [[-1,1], [1,2,-3], [-2,3], [3,2,-4]])
    e = eigs(Ltemp.reshape(m**2, m**2).T, k = 1, which = 'LM', return_eigenvectors = False)
    Gamma /= np.real(e)**0.5 # Make the leading eigenvalue == 1

    Ltemp = ncon([Lambda, Gamma, Lambda, Gamma.conj()], [[-1,1], [1,2,-3], [-2,3], [3,2,-4]])
    Rtemp = ncon([Gamma, Lambda, Gamma.conj(), Lambda],[[-1,2,1], [1,-3], [-2,2,3], [3,-4]])

    l = find_lr(Ltemp.reshape(m**2, m**2).T, m)
    r = find_lr(Rtemp.reshape(m**2, m**2), m)

    L = linalg.cholesky(l, check_finite = False)
    R = linalg.cholesky(r, check_finite = False).conj().T

    Linv = linalg.inv(L.conj())

    Rinv = linalg.inv(R)
    u, Lambda, vt = linalg.svd(ncon([L.conj(), Lambda, R],[[-1,1],[1,2],[2,-2]]), full_matrices=False)
    Lambda /= linalg.norm(Lambda); Lambda = np.diag(Lambda)
    Gamma = ncon([vt,Rinv,Gamma,Linv,u],[[-1,1],[1,2],[2,-2,3],[3,4],[4,-3]])
    Gamma /= ncon([Lambda, Gamma, Lambda, Gamma.conj()],[[4,1],[1,2,-1],[4,3],[3,2,-2]])[0,0]**0.5 # make the leading eigenvalue == 1
    
    return Lambda, Gamma

def toSymmetricGauge(AL):
    Lambda, Gamma = toLambdaGamma(AL)
    Lambda, Gamma = toCanonical(Lambda, Gamma)
    sqLambda = linalg.sqrtm(Lambda)
    A = ncon([sqLambda, Gamma, sqLambda],[[-1,1],[1,-2,2],[2,-3]])
    return A, Lambda  # Lambda is the leading eigenvector

def LeftGaugefixed(A, l, r):
    m = A.shape[0]
    d = A.shape[1]
    temp = ncon([linalg.sqrtm(l), A.conj()],[[-2, 1],[1, -3, -1]]).reshape(m, m*d)
    Vl = linalg.null_space(temp)[:, range(m*(d-1))]
    Vl = np.linalg.svd(Vl, full_matrices=False)[0].reshape(m,d,m*(d-1))
    return Vl

def getlr(A):
    m = A.shape[0]
    TM = ncon([A, A.conj()], [[-1,1,-3], [-2,1,-4]]).reshape(m**2, m**2)
    e, r = eigs(TM, k = 1, which = 'LM')
    e, l = eigs(TM.T, k = 1, which = 'LM') 
    r = r.reshape(m, m)
    l = l.reshape(m, m)
    r = 0.5*(r+r.conj().T)
    l = 0.5*(l+l.conj().T)

    r = truncate(r)
    l = truncate(l)

    tr = ncon([l,r],[[1,2],[1,2]])
    l/=tr**0.5
    r/=tr**0.5
    # print(np.linalg.cond(l))
    # print(np.linalg.cond(r))
    return l, r

def getTME(A, A0):
    m = A.shape[0]
    TM = ncon([A, A0.conj()], [[-1, 1,-3],[-2, 1,-4]]).reshape(m**2, m**2)
    return np.abs(eigs(TM, k = 1, which = 'LM',  return_eigenvectors = False))

def getEE(C):
    ee1 = 0
    ee2 = 0
    for x in C:
        ee1 += -(x**2 * np.log2(x**2))
        ee2 += -(x**2 * np.log(x**2))
    return [ee1, ee2]

def getCC(A, Lambda):
    # sZ =  1/2 * np.array([[1.0, 0], [0, -1.0]])
    sZ = np.array([[1.0, 0], [0, -1.0]]) # for ising
    sI =  np.array([[1.0, 0], [0, 1.0]])
    meaa, meab = np.kron(sI,sZ), np.kron(sZ,sI)
    norm = ncon([Lambda, Lambda], [[1,2],[1,2]])
    a = ncon([Lambda, A, meaa, A.conj(), Lambda], [[5,6],[5,2,7],[2,4],[6,4,8],[7,8]])/norm
    b = ncon([Lambda, A, meab, A.conj(), Lambda], [[5,6],[5,2,7],[2,4],[6,4,8],[7,8]])/norm
    cc = 0.5*(a-b)
    mz = 0.5*(a+b)
    return cc, mz

# def getMz(A, Lambda):
#     # sZ =  np.array([[1.0, 0], [0, -1.0]]) # Pauli for the ising case
#     # mZ = ncon([Lambda, A, sZ, A.conj(), Lambda], [[5,6],[5,2,7],[2,4],[6,4,8],[7,8]]) / ncon([Lambda, Lambda], [[1,2],[1,2]])
#     return mZ

def tdvpSG(A0, W, dt, numiter, backup_step = 200, bicg_tol = 1e-12, RK4 = True, filename = "Unnamed"):

    def getLWaC25(LWa, YLa, AL, L, R):
        m = AL.shape[0]
        def Lcontract(v):

            v = v.reshape(m,m)
            # LWa_TL = ncon([v, TL],[[1,2],[1,-1,2,-2]])
            LWa_TL = ncon([v, AL, AL.conj()],[[1,2],[1,3,-1],[2,3,-2]])
            # LWa_P = ncon([v, R], [[1,2],[1,2]]) * np.eye(m)
            LWa_P = ncon([v, R], [[1,2],[1,2]]) * L
            return v.flatten() - LWa_TL.flatten() + LWa_P.flatten()

        LeftOp = LinearOperator((m**2, m**2), matvec=Lcontract, dtype = np.cdouble)

        # B = YLa - ncon([YLa, R], [[1,2],[1,2]])*np.eye(m)
        B = YLa - ncon([YLa, R], [[1,2],[1,2]]) * L
        LWa_temp, is_conv = bicgstab(LeftOp, B.flatten(), maxiter = 100000, tol=bicg_tol) # x0=LWa.flatten()?
        if is_conv != 0:
            print("bicgstab didn't converge : %d"%is_conv)
            if is_conv<0:
                print("bicgstab breakdown.")
                exit(1)
        # LWa_temp = np.real(LWa_temp).reshape(m, m)
        LWa_temp = LWa_temp.reshape(m, m)

        return LWa_temp

    def getRWaC25(RWa, YRa, AR, L, R):
        m = AR.shape[0]
        def Rcontract(v):
            v = v.reshape(m,m)
            # RWa_TR = ncon([TR, v],[[-1,1,-2,2], [1,2]])
            RWa_TR = ncon([v, AR, AR.conj()],[[1,2], [-1,3,1], [-2,3,2]])
            # RWa_P = ncon([L, v], [[1,2],[1,2]])*np.eye(m)
            RWa_P = ncon([L, v], [[1,2],[1,2]]) * R
            return v.flatten() - RWa_TR.flatten() + RWa_P.flatten()

        RightOp = LinearOperator((m**2, m**2), matvec=Rcontract, dtype = np.cdouble)

        # B = YRa - ncon([L, YRa], [[1,2],[1,2]]) * np.eye(m)
        B = YRa - ncon([L, YRa], [[1,2],[1,2]]) * R
        RWa_temp, is_conv = bicgstab(RightOp, B.flatten(), maxiter = 100000, tol=bicg_tol) # x0=RWa.flatten()?
        if is_conv != 0:
            print("bicgstab didn't converge : %d"%is_conv)
            if is_conv<0:
                print("bicgstab breakdown.")
                exit(1)
        # RWa_temp = np.real(RWa_temp).reshape(m, m)
        RWa_temp = RWa_temp.reshape(m, m)

        return RWa_temp

    def getLW(A, l, r):
        m = A.shape[0]
        dw = W.shape[0]
        YL = [np.zeros([m,m], dtype = np.cdouble) for i in range(dw)]
        # LW_ = LW.copy()
        LW_ = [np.zeros([m,m], dtype = np.cdouble) for i in range(dw)]
        # LW_[dw-1] = np.eye(m)
        LW_[dw-1] = l
        R = r
        for a in range(dw-2, -1, -1):
            for b in range(a+1, dw):
                YL[a] += ncon([LW_[b], A, W[b,a], A.conj()],[[1,2], [1,4,-1], [4,5], [2,5,-2]])
            if W[a, a, 0, 0] == 0:
                LW_[a] = YL[a]
            elif W[a, a, 0, 0] == 1:
                LW_[a] = getLWaC25(LW_[a], YL[a], A, l, R)
        
        return np.asarray(LW_)#, ncon([YL[0], R], [[1,2],[1,2]])

    def getRW(A, l, r):
        m = A.shape[0]
        dw = W.shape[0]
        YR = [np.zeros([m,m], dtype = np.cdouble) for i in range(dw)]
        # RW_ = RW.copy()
        RW_ = [np.zeros([m,m], dtype = np.cdouble) for i in range(dw)]
        # RW_[0] = np.eye(m)
        RW_[0] = r
        L = l
        for a in range(1, dw):
            for b in range(a-1, -1, -1):
                YR[a] += ncon([RW_[b], A, W[a,b], A.conj()],[[1,2], [-1,4,1], [4,5], [-2,5,2]])
            if W[a, a, 0, 0] == 0:
                RW_[a] = YR[a]
            elif W[a, a, 0, 0] == 1:
                RW_[a] = getRWaC25(RW_[a], YR[a], A, L, r)

        return np.asarray(RW_) #, ncon([L, YR[-1]], [[1,2],[1,2]])

    def getB(A):
        
        l, r = getlr(A)
        Vl = LeftGaugefixed(A, l, r)
        L, R = getLW(A, l, r), getRW(A, l, r)
        LAOR = ncon([L,A,W,R],[[2,1,-1],[1,5,4],[2,3,5,-2],[3,4,-3]])
        l, r = linalg.sqrtm(linalg.inv(l)), linalg.inv(r)
        lVVl = ncon([l, Vl.conj(), Vl, l], [[-1,1],[1,-2,2],[3,-3,2],[-4,3]])
        B = ncon([LAOR, lVVl, r],[[1,2,3],[1,2,-2,-1],[3,-3]])
        return B
    
    m = A0.shape[0]
    A = np.copy(A0)
    A0, _ = toSymmetricGauge(A0)
    Pts = []; ees = []; ccs = []; mzs = []; As = []; Lambdas = [];

    if RK4:
        # Note that we cannot change the gauge for uMPS until whole RK4 process is done
        for ite in range(numiter):

            A, Lambda = toSymmetricGauge(A)
            print("Iteration : %d, cond. num : %.2e"%(ite, np.linalg.cond(Lambda)))
            As.append(A); Lambdas.append(Lambda);
            ees.append(getEE(np.diag(Lambda))[1])
            Pts.append(-np.log(getTME(A, A0)))
            cc, mz = getCC(A, Lambda)
            ccs.append(cc)
            mzs.append(mz)
            if ite%backup_step == 0 and ite != 0:
                np.save("Backup_As_"+filename,  np.asarray(As))
                np.save("Backup_Lambdas_"+filename,  np.asarray(Lambdas))
                np.save("Backup_datas_"+filename, np.asarray([ees, ccs, Pts, mzs], dtype=object))

            B1 = getB(A); A1 = A-1j*(1/2)*B1*dt
            B2 = getB(A1); A2 = A-1j*(1/2)*B2*dt
            B3 = getB(A2); A3 = A-1j*B3*dt
            B4 = getB(A3)
            A = A - 1j * (1/6) * (B1 + 2*B2 + 2*B3 + B4) * dt
    
        return np.asarray(As), np.asarray(Lambdas), np.asarray([ees, ccs, Pts, mzs], dtype=object)

    else :
        for ite in range(numiter):
            # print(ite)
            A, Lambda = toSymmetricGauge(A)
            print(np.linalg.cond(Lambda))

            B = getB(A)
            A = A - B*dt
            # print("|B| = %.5f"%linalg.norm(B))
            # if linalg.norm(B) < 1:
            #     break
        return A
