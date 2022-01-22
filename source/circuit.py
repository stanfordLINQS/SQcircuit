# Libraries:
from elements import *
from PhysicsConstants import *
import numpy as np
import qutip as q

import scipy.special
import scipy.sparse
from scipy.linalg import sqrtm, block_diag
import copy
import collections


class Circuit:
    """
    SQcircuit class gets superconducting quantum circuit and calculates the following circuit properties:
    -- eigenvalues and eigenvectors
    -- phase space representation eigenvectors
    -- matrix elements of the circuit
    -- decay rates
    """

    def __init__(self, circuitParam: dict):

        """
        inputs:
            -- circuitParam: a dictionary that contains the circuit properties at each edge or
                            branch of the circuit.
        """

        self.circuitParam = collections.defaultdict(lambda: [], copy.deepcopy(circuitParam))

        # number of nodes
        self.n = max(max(self.circuitParam))

        # the inverse of transformation of coordinates for charge operators
        self.R = np.zeros((self.n, self.n))

        # the inverse of transformation of coordinates for flux operators
        self.S = np.zeros((self.n, self.n))

        # transform the Hamiltonian of the circuit
        self.transformHamil()

        # truncation numbers for each mode
        self.m = []

        # list of charge operators (self.n)
        self.chargeOpList = []
        # cross multiplication of charge operators as list
        self.chargeByChargeList = []
        # list of number operators (self.n)
        self.numOpList = []
        # LC part of the Hamiltonian
        self.HLC = q.Qobj()
        # List of exponential part of the Josephson Junction cosine
        self.HJJExpList = []
        # List of square root of exponential part of
        self.HJJExpRootList = []
        # sin(phi/2) operator related to each JJ for quasi-particle Loss
        self.qpSinList = []
        # external fluxes of the circuit
        self.extFlux = {}
        # eigenvalues of the circuit
        self.HamilEigVal = []
        # eigenvectors of the circuit
        self.HamilEigVec = []

    @staticmethod
    def elementModel(elementList: list, model):
        """
            get the list of element with specific model from the list of elements.
            inputs:
                -- elementList: a list of objects from Capacitor, Inductor, and JJ class.
                -- model: model of the element( can be Capacitor, Inductor, or JJ)
            outputs:
                -- modelList: list of element with specified model.
        """
        return [el for el in elementList if isinstance(el, model)]

    def getMatC(self):
        """ Get the capacitance matrix from circuit parameters.
        output:
            -- cMat: capacitance matrix (self.n,self.n)
        """

        cMat = np.zeros((self.n, self.n))

        for edge in self.circuitParam.keys():
            # i1 and i2 are the nodes of the edge
            i1, i2 = edge

            # list of capacitors of the edge.
            capList = self.elementModel(self.circuitParam[edge], Capacitor)

            # summation of the capacitor values.
            cap = sum(list(map(Capacitor.value, capList)))

            if i1 != 0 and i2 == 0:
                cMat[i1 - 1, i1 - 1] += cap
            elif i1 == 0 and i2 != 0:
                cMat[i2 - 1, i2 - 1] += cap
            else:
                cMat[i1 - 1, i2 - 1] = - cap
                cMat[i2 - 1, i1 - 1] = - cap
                cMat[i1 - 1, i1 - 1] += cap
                cMat[i2 - 1, i2 - 1] += cap

        return cMat

    def getMatL(self):
        """ Get the inductance matrix from circuit parameters.
        output:
            -- cMat: inductance matrix (self.n,self.n)
        """

        lMat = np.zeros((self.n, self.n))

        for edge in self.circuitParam.keys():
            # i1 and i2 are the nodes of the edge
            i1, i2 = edge

            # list of inductors of the edge
            indList = self.elementModel(self.circuitParam[edge], Inductor)

            # summation of the inductor values.
            x = np.sum(1 / np.array(list(map(Inductor.value, indList))))

            if i1 != 0 and i2 == 0:
                lMat[i1 - 1, i1 - 1] += x
            elif i1 == 0 and i2 != 0:
                lMat[i2 - 1, i2 - 1] += x
            else:
                lMat[i1 - 1, i2 - 1] = -x
                lMat[i2 - 1, i1 - 1] = -x
                lMat[i1 - 1, i1 - 1] += x
                lMat[i2 - 1, i2 - 1] += x

        return lMat

    def getMatW(self):
        """Get the w matrix which contains the linear combination of
        the flux coordinates in Josephson Junction cosine without transformation
        of coordinates.
        output:
            -- wMat: W matrix(linear combination of the fluxes in the
                    JJ cosine (n_J,self.n)
        """

        wMat = []
        for edge in self.circuitParam.keys():
            # i1 and i2 are the nodes of the edge
            i1, i2 = edge

            # list of Josephson Junction of the edge.
            JJList = self.elementModel(self.circuitParam[edge], Junction)

            if len(JJList) != 0:
                w = [0] * (self.n + 1)

                if i1 == 0 or i2 == 0:
                    w[i1 + i2] += 1
                else:
                    w[i1] += 1
                    w[i2] -= 1

                wMat.append(w[1:])

        wMat = np.array(wMat)

        return wMat

    def transform1(self):
        """
        First transformation of the coordinates that simultaneously diagonalizes
        the capacitance and inductance matrices.

        output:
            --  lDiag: diagonalized inductance matrix (self.n,self.n)
            --  cInvDiag: diagonalized inverse of capacitance matrix (self.n,self.n)
            --  R1: transformation of charge operators (self.n,self.n)
            --  S1: transformation of flux operators (self.n,self.n)
        """

        cMat = self.getMatC()
        lMat = self.getMatL()
        cMatInv = np.linalg.inv(cMat)

        cMatRoot = sqrtm(cMat)
        cMatRootInv = np.linalg.inv(cMatRoot)
        lMatRoot = sqrtm(lMat)

        V, D, U = np.linalg.svd(lMatRoot @ cMatRootInv)

        # find the zero singular values
        singLoc = []

        # the case that there is not any inductor in the circuit
        if np.max(D) == 0:
            D = np.diag(np.eye(self.n))
            singLoc = list(range(0, self.n))
        else:
            for i in range(self.n):
                if D[i] / np.max(D) < 1e-6:
                    singLoc.append(i)
            D[singLoc] = np.max(D)

        # build S1 and R1 matrix
        S1 = cMatRootInv @ U.T @ np.diag(np.sqrt(D))
        R1 = np.linalg.inv(S1).T

        cInvDiag = R1.T @ cMatInv @ R1
        lDiag = S1.T @ lMat @ S1

        lDiag[singLoc, singLoc] = 0

        return lDiag, cInvDiag, S1, R1

    def transform2(self, omega: np.array, S1: np.array):
        """
        Second transformation of the coordinates that transforms the subspace of
        the charge operators which are defined in the charge basis in order
        to have the Bloch wave vectors in the cartesian direction.
        output:
            --  R2: Second transformation of charge operators (self.n,self.n)
            --  S2: Second transformation of flux operators (self.n,self.n)
        """

        # apply the first transformation on w and get the charge basis part
        wTrans1 = self.getMatW() @ S1
        wQ = wTrans1[:, omega == 0]
        # number of operators represented in charge bases
        nq = wQ.shape[1]

        # if we need to represent an operator in charge basis
        if nq != 0:

            # normalizing the wQ vectors(each row is a vector)
            wQ_norm = wQ / np.linalg.norm(wQ, axis=1).reshape(wQ.shape[0], 1)

            # list of indices of w vectors that are independent
            indList = [0]
            j = 1
            while len(indList) != nq:
                # inner product of the jth w with selected wQ( indList)
                iner = np.abs(np.sum(wQ_norm[indList, :] * wQ_norm[j, :], 1))
                # check if we found a new w that is not parallel with selected w
                if np.max(iner) <= 1 - 1e-12:
                    indList.append(j)
                j += 1

            # the second S and R matrix are:
            S2 = block_diag(np.eye(self.n - nq), np.linalg.inv(wQ[indList, :]))
            R2 = np.linalg.inv(S2.T)

        else:
            S2 = np.eye(self.n, self.n)
            R2 = S2

        return S2, R2

    def transform3(self):
        """ Third transformation of the coordinates that scales the modes.
        output:
            --  R3: Third transformation of charge operators (self.n,self.n)
            --  S3: Third transformation of flux operators (self.n,self.n)
        """

        S3 = np.eye(self.n)

        for j in range(self.n):

            # for the charge basis
            if self.omega[j] == 0:
                s = np.max(np.abs(self.wTrans[:, j]))
                if s != 0:
                    for i in range(len(self.wTrans[:, j])):
                        # check if abs(A[i,j]/s is either zero or one with 1e-11 accuracy
                        if abs(self.wTrans[i, j] / s) >= 1e-11 and abs(abs(self.wTrans[i, j] / s) - 1) >= 1e-11:
                            raise ValueError("This solver cannot solve your circuit.")
                        if abs(self.wTrans[i, j] / s) <= 1e-11:
                            self.wTrans[i, j] = 0

                    S3[j, j] = 1 / s

                # correcting the cInvRotated values
                for i in range(self.n):
                    if i == j:
                        self.cInvDiag[i, j] = self.cInvDiag[i, j] * s ** 2
                    else:
                        self.cInvDiag[i, j] = self.cInvDiag[i, j] * s
            # for harmonic modes
            else:
                # note: alpha here is absolute value of alpha( alpha is pure imaginary)
                # alpha for j-th mode
                alpha = np.abs(2 * np.pi / Phi0 * np.sqrt(hbar / 2 * np.sqrt(
                    self.cInvDiag[j, j] / self.lDiag[j, j])) * self.wTrans[:, j])
                self.wTrans[:, j][alpha < 1e-11] = 0
                if np.max(alpha) > 1e-11:
                    # find the coefficient in wTrans for j-th mode that has maximum alpha
                    s = np.abs(self.wTrans[np.argmax(alpha), j])
                    # scale that mode with s
                    self.wTrans[:, j] = self.wTrans[:, j] / s
                    self.cInvDiag[j] *= s ** 2
                    self.lDiag[j] /= s ** 2
                    S3[j, j] = 1 / s

        R3 = np.linalg.inv(S3.T)

        return S3, R3

    def transformHamil(self):
        """
        transform the Hamiltonian of the circuit that can be expressed
        in charge and Fock bases
        """

        # get the first transformation:
        self.lDiag, self.cInvDiag, self.S1, self.R1 = self.transform1()
        # second transformation

        # natural frequencies of the circuit(zero for modes in charge basis)
        self.omega = np.sqrt(np.diag(self.cInvDiag) * np.diag(self.lDiag))

        # get the second transformation:
        self.S2, self.R2 = self.transform2(self.omega, self.S1)

        # apply the second transformation on self.cInvDiag
        self.cInvDiag = self.R2.T @ self.cInvDiag @ self.R2

        # get the transformed W matrix
        self.wTrans = self.getMatW() @ self.S1 @ self.S2

        # scaling the modes
        self.S3, self.R3 = self.transform3()

        # The final transformations are:
        self.S = self.S1 @ self.S2 @ self.S3
        self.R = self.R1 @ self.R2 @ self.R3

        print("Natural frequencies of the circuit:")
        print(self.omega)
        print("W transformed matrix:")
        print(self.wTrans)

    def setTruncationNumbers(self, truncNum: list):
        """set the truncation numbers for each mode
        input:
            -- truncNum: a list that contains the truncation number for each mode (self.n)
        """
        # set the truncation number for the fock state
        error1 = "The input must be be a python list"
        assert isinstance(truncNum, list), error1
        error2 = "The number of modes(length of the input) must be equal to the number of nodes"
        assert len(truncNum) == self.n, error2
        self.m = truncNum

        self.chargeOpList, self.numOpList, self.chargeByChargeList = self.buildOpMemory(self.lDiag, self.cInvDiag,
                                                                                        self.omega, self.m, self.n)

        self.HLC = self.getLCHamil(self.cInvDiag, self.omega, self.chargeByChargeList, self.numOpList)

        self.HJJExpList, self.HJJExpRootList = self.getHJJExp(self.cInvDiag, self.lDiag, self.omega,
                                                              self.wTrans, self.m, self.n)

    def setExternalFluxes(self, externalFluxes: dict):
        """set the external fluxes for each Josephson Junction
        input:
            -- externalFluxes: a dictionary that contains the external flux
            at each edge
        """
        assert isinstance(externalFluxes, dict), "The input must be be a python dictionary"
        self.extFlux = externalFluxes

    def buildOpMemory(self, lDiag: np.array, cInvDiag: np.array, omega: np.array, m: list, n: int):
        """
        build the charge operators, number operators, and cross multiplication of
        charge operators.
        inputs:
            -- lDiag: diagonalized inductance matrix (self.n,self.n)
            -- cInvDiag: diagonalized inverse of capacitance matrix (self.n,self.n)
            -- omega: natural frequencies of the circuit (self.n)
            -- m: list of truncation numbers (self.n)
            -- n: number of circuit nodes
        outputs:
            -- chargeOpList : list of charge operators (self.n)
            -- chargeByChargeList : cross multiplication of charge operators as list
            -- numOpList : list of number operators (self.n)
        """

        # list of charge operators in their own mode basis
        # (tensor product of other modes are not applied yet!)

        chargeOpList = []
        numOpList = []
        chargeByChargeList = []

        QList = []
        for i in range(n):
            if omega[i] == 0:
                Q0 = (2 * e / np.sqrt(hbar)) * q.charge((m[i] - 1) / 2)
            else:
                coef = -1j * np.sqrt(1 / 2 * np.sqrt(lDiag[i, i] / cInvDiag[i, i]))
                Q0 = coef * (q.destroy(m[i]) - q.create(m[i]))
            QList.append(Q0)

        # list of number operators in their own mode basis
        # (tensor product of other modes are not applied yet!)
        nList = []
        for i in range(n):
            if omega[i] == 0:
                num0 = q.charge((m[i] - 1) / 2)
            else:
                num0 = q.num(m[i])
            nList.append(num0)

        for i in range(n):
            chargeRowList = []
            for j in range(n):

                # find the appropriate charge and number operator for first mode
                if j == 0 and i == 0:
                    Q2 = QList[j] * QList[j]
                    Q = QList[j]
                    num = nList[j]

                    # Tensor product the charge with I for other modes
                    for k in range(n - 1):
                        Q2 = q.tensor(Q2, q.qeye(m[k + 1]))
                    chargeRowList.append(Q2)

                elif j == 0 and i != 0:
                    I = q.qeye(m[j])
                    Q = I
                    num = I

                # find the rest of the modes
                elif j != 0 and j < i:
                    I = q.qeye(m[j])
                    Q = q.tensor(Q, I)
                    num = q.tensor(num, I)

                elif j != 0 and j == i:
                    Q2 = q.tensor(Q, QList[j] * QList[j])
                    Q = q.tensor(Q, QList[j])
                    num = q.tensor(num, nList[j])

                    # Tensor product the charge with I for other modes
                    for k in range(n - j - 1):
                        Q2 = q.tensor(Q2, q.qeye(m[k + j + 1]))
                    chargeRowList.append(Q2)

                elif j > i:
                    QQ = q.tensor(Q, QList[j])

                    # Tensor product the QQ with I for other modes
                    for k in range(n - j - 1):
                        QQ = q.tensor(QQ, q.qeye(m[k + j + 1]))
                    chargeRowList.append(QQ)

                    I = q.qeye(m[j])
                    Q = q.tensor(Q, I)
                    num = q.tensor(num, I)

            chargeOpList.append(Q)
            numOpList.append(num)
            chargeByChargeList.append(chargeRowList)

        return chargeOpList, numOpList, chargeByChargeList

    def getLCHamil(self, cInvDiag: np.array, omega: np.array, chargeByChargeList: list, numOpList: list):
        """
        get the LC part of the Hamiltonian
        inputs:
            --  cInvDiag: diagonalized inverse of capacitance matrix (self.n,self.n)
            -- chargeByChargeList: cross multiplication of charge operators as list
            -- list of number operators (self.n)
            -- omega: natural frequencies of the circuit (self.n)
        outputs:
            -- HLC: LC part of the Hamiltonian (qutip Object)
        """

        HLC = 0

        for i in range(self.n):
            # we write j in this form because of chargeByChargeList shape
            for j in range(self.n - i):
                if j == 0:
                    if self.omega[i] == 0:
                        HLC += 1 / 2 * cInvDiag[i, i] * chargeByChargeList[i][j]
                    else:
                        HLC += omega[i] * numOpList[i]
                elif j > 0:
                    if cInvDiag[i, i + j] != 0:
                        HLC += cInvDiag[i, i + j] * chargeByChargeList[i][j]

        return HLC

    @staticmethod
    def chargeDisp(N: int):
        """
        return charge displacement operator with size N.
        input:
            -- N: size of the Hilbert Space
        output:
            -- d: charge displace ment operator( qutip object)
        """
        d = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if j - 1 == i:
                    d[i, j] = 1
        d = q.Qobj(d)
        d = d.dag()

        return d

    def getHJJExp(self, cInvDiag: np.array, lDiag: np.array, omega: np.array, wTrans: np.array, m: list, n: int):
        """
        Each cosine potential of the Josephson Junction can be written as summation of two
        exponential terms,cos(x)=(exp(ix)+exp(-ix))/2. This function returns the quantum
        operators for only one exponential term.
        inputs:
            -- lDiag: diagonalized inductance matrix (self.n,self.n)
            -- cInvDiag: diagonalized inverse of capacitance matrix (self.n,self.n)
            -- omega: natural frequencies of the circuit (self.n)
            -- wTrans: transformed W matrix (nJ,self.n)
            -- m: list of truncation numbers (self.n)
            -- n: number of circuit nodes
        outputs:
            -- HJJExpList: List of exponential part of the Josephson Junction cosine (nJ)
            -- HJJExpHalfList: List of square root of exponential part of
                               the Josephson Junction cosine (nJ)
        """

        HJJExpList = []
        HJJExpRootList = []

        # number of Josephson Junctions
        nJ = wTrans.shape[0]

        H = 0
        # for calculating sin(phi/2) operator for quasi-particle loss decay rate
        H2 = 0

        for i in range(nJ):

            # tensor multiplication of displacement operator for JJ Hamiltonian
            for j in range(n):
                if j == 0 and omega[j] == 0:
                    if wTrans[i, j] == 0:
                        I = q.qeye(m[j])
                        H = I
                        H2 = I
                    elif wTrans[i, j] > 0:
                        d = self.chargeDisp(m[j])
                        H = d
                        # not correct just to avoid error:
                        H2 = d
                    else:
                        d = self.chargeDisp(m[j])
                        H = d.dag()
                        # not correct just to avoid error:
                        H2 = d

                elif j == 0 and omega[j] != 0:
                    alpha = 2 * np.pi / Phi0 * 1j * np.sqrt(
                        hbar / 2 * np.sqrt(cInvDiag[j, j] / lDiag[j, j])) * wTrans[i, j]
                    H = q.displace(m[j], alpha)
                    H2 = q.displace(m[j], alpha / 2)

                if j != 0 and omega[j] == 0:
                    if wTrans[i, j] == 0:
                        I = q.qeye(m[j])
                        H = q.tensor(H, I)
                        H2 = q.tensor(H2, I)
                    elif wTrans[i, j] > 0:
                        d = self.chargeDisp(m[j])
                        H = q.tensor(H, d)
                    else:
                        d = self.chargeDisp(m[j])
                        H = q.tensor(H, d.dag())

                elif j != 0 and omega[j] != 0:
                    alpha = 2 * np.pi / Phi0 * 1j * np.sqrt(
                        hbar / 2 * np.sqrt(cInvDiag[j, j] / lDiag[j, j])) * wTrans[i, j]
                    H = q.tensor(H, q.displace(m[j], alpha))
                    H2 = q.tensor(H2, q.displace(m[j], alpha / 2))

            HJJExpList.append(H)
            HJJExpRootList.append(H2)

        return HJJExpList, HJJExpRootList

    def getJJHamil(self, HJJExpList: list, HJJExpRootList: list, fluxExt: dict):
        """
        get the Josephson Junction part of the Hamiltonian, cos(phi), for each Josephson Junction.
        It also returns the sin(phi/2) for quasi-particle loss calculation.
        inputs:
            -- HJJExpList: List of exponential part of the Josephson Junction cosine (nJ)
            -- HJJExpRootList: List of square root of exponential part of the Josephson Junction cosine (nJ)
            -- fluxExt:  A dictionary that contains the external flux at each edge
        outputs:
            -- HJJ: Josephson Junctions part of the Hamiltonian(qutip operator)
            -- HJJSinHalfList: A list that contains the sin(phi/2) part of Hamiltonian for each JJ (nJ)
        """

        # List of each JJ Hamiltonian
        HJJList = []

        # list of sin(phi/2) of each JJs
        HJJSinHalfList = []

        # count when we address any JJ( when EJ is not None)
        i = 0

        for edge in self.circuitParam:

            # list of Josephson Junction of the edge.
            JJList = self.elementModel(self.circuitParam[edge], Junction)
            if len(JJList) == 0:
                continue

            EJ = list(map(Junction.value, JJList))
            # Parallel JJ case
            if len(EJ) > 1:

                H = 0
                phi = fluxExt.get(edge, []) + fluxExt.get((edge[1], edge[0]), [])
                phi = phi + [0] * (len(EJ) - len(phi))

                for j in range(len(EJ)):
                    H += np.exp(1j * phi[j]) * EJ[j] / 2 * HJJExpList[i]

                H = H + H.dag()
                # needed to be implemented 
                H2 = 0

            # single JJ case
            else:
                # return zero if external flux is not specified for that edge
                phi = fluxExt.get(edge, 0) + fluxExt.get((edge[1], edge[0]), 0)

                H = np.exp(1j * phi) * EJ[0] / 2 * HJJExpList[i]
                H = H + H.dag()

                # sin(phi/2) for the quasi-particle decay rate
                H2 = np.exp(1j * phi / 2) * HJJExpRootList[i]
                H2 = q.Qobj(H2)
                H2 = (H2.dag() - H2) / 2j

            HJJList.append(H)
            HJJSinHalfList.append(H2)
            i += 1

        HJJ = sum(HJJList)

        return HJJ, HJJSinHalfList

    def run(self, numEig: int):
        """
        calculate the Hamiltonian of the circuit and get the eigenvalue and eigenvectors of the circuit up
        to specified number of eigenvalues( reducing the numEig can speed up the eigen solver)
        inputs:
            -- numEig: int variable that specifies the number of eigenvalues that eigen solver returns
        output:
            -- eigenvaluesSorted: the eigen values of the Hamiltonian (eigNum)
            -- eigenVectorsSorted: a list of qutip operators that contains the eigenvectors (eigNum)
        """

        assert len(self.m) != 0, "Please specify the truncation number for each mode."
        assert isinstance(numEig, int), "The numEig( number of eigenvalues) should be an integer."

        HJJ, self.qpSinList = self.getJJHamil(self.HJJExpList, self.HJJExpRootList, self.extFlux)

        H = -HJJ + self.HLC

        # get the data out of qutip variable and use scipy eigen solver which is faster than
        # qutip eigen solver( I tested this experimentally)
        eigenValues, eigenVectors = scipy.sparse.linalg.eigs(H.data, numEig, which='SR')
        # the output of eigen solver is not sorted
        eigenValuesSorted = np.sort(eigenValues.real)
        sortArg = np.argsort(eigenValues)
        eigenVectorsSorted = [q.Qobj(eigenVectors[:, ind], dims=[self.m, len(self.m) * [1]])
                              for ind in sortArg]

        # store the eigenvalues and eigenvectors of the circuit Hamiltonian
        self.HamilEigVal = eigenValuesSorted
        self.HamilEigVec = eigenVectorsSorted

        return eigenValuesSorted, eigenVectorsSorted

    ###############################################
    # Methods that calculate circuit properties
    ###############################################

    def coordinateTransformation(self, opType: str):
        """
        returns the transformation of the coordinates for each type of operators( either charge operators or flux
        operators)
        inputs:
            -- opType: the type of the operator that can be either "charge" or "flux".
        outputs:
            -- transCoord: transformation of the charge or flux node operator (self.n, self.n)
        """
        if opType == "charge" or opType == "Charge":
            return np.linalg.inv(self.R)
        elif opType == "flux" or opType == "Flux":
            return np.linalg.inv(self.S)
        else:
            raise ValueError(" The input must be either \"charge\" or \"flux\".")

    def hamiltonian(self, part="all"):
        """
        returns the transformed hamiltonian of the circuit for specified part that can be LC, JJ, or both parts
        of the Hamiltonian.
        inputs:
            -- part: the specific part of the Hamiltonian( can be "LC", "JJ", or "all")
        """
        assert len(self.m) != 0, "Please specify the truncation number for each mode."

        if part == "LC" or part == "lc":
            return self.HLC
        elif part == "JJ" or part == "jj":
            HJJ, _ = self.getJJHamil(self.HJJExpList, self.HJJExpRootList, self.extFlux)
            return HJJ
        elif part == "all":
            HJJ, _ = self.getJJHamil(self.HJJExpList, self.HJJExpRootList, self.extFlux)
            return -HJJ + self.HLC
        else:
            raise ValueError("The input must be either, \"LC\". \"JJ\", or \"all\".")

    def operator(self, opType, node):
        pass

    def tensorToModes(self, tensorIndex: int):
        """
        decomposes the tensor product space index to each mode indices. For example index 5 of the tensor
        product space can be decomposed to [1,0,1] modes if the truncation number for each mode is 2.
        inputs:
            -- tensorIndex: Index of tensor product space
        outputs:
            -- indList: a list of mode indices (self.n)
        """

        # i-th mP element is the multiplication of the self.m elements until its i-th element
        mP = []
        for i in range(self.n - 1):
            if i == 0:
                mP.append(self.m[-1])
            else:
                mP = [mP[0] * self.m[-1 - i]] + mP

        indList = []
        indexP = tensorIndex
        for i in range(self.n):
            if i == self.n - 1:
                indList.append(indexP)
                continue
            indList.append(int(indexP / mP[i]))
            indexP = indexP % mP[i]

        return indList

    def eigVecPhaseSpace(self, eigInd: int, phiList: list):
        """
        gets the eigenvectors in the phase space representation.
        inputs:
            -- eigInd: the index of the eigenvector
            -- phaseList: list of phases for all nodes
        outputs:
            -- state: the eigenvector represented in phase space
        """

        assert isinstance(eigInd, int), "The eigInd( eigen index) should be an integer."

        # The total dimension of the circuit Hilbert Space
        netDimension = np.prod(self.m)

        state = 0

        for i in range(netDimension):

            # decomposes the tensor product space index (i) to each mode indices as a list
            indList = self.tensorToModes(i)

            term = self.HamilEigVec[eigInd][i][0, 0]

            for mode in range(self.n):

                # mode number related to that node
                n = indList[mode]

                # For charge basis
                if self.omega[mode] == 0:
                    term *= 1 / np.sqrt(2 * np.pi) * np.exp(1j * phiList[mode] * n)
                # For harmonic basis
                else:
                    x0 = np.sqrt(hbar * np.sqrt(self.cInvDiag[mode, mode] / self.lDiag[mode, mode]))

                    coef = 1 / np.sqrt(np.sqrt(np.pi) * 2 ** n * scipy.special.factorial(n) * x0/Phi0)

                    term *= coef * np.exp(-(phiList[mode] * Phi0 / x0) ** 2 / 2) * \
                            scipy.special.eval_hermite(n, phiList[mode] * Phi0 / x0)

            state += term

        return state

    # def setDieTanLoss(self, DieTanList):
    #     self.dieTanList = DieTanList;
    #
    # def setQuasiparticleFraction(self, x_qpList):
    #     self.x_qpList = x_qpList
    #
    # def setTemperature(self, T):
    #     self.T = T
    #
    # def decayRateProcess(self, state1, state2, mode='all'):
    #     """function that calculate the effective decay rates for
    #     specific process of |state1> and |state2>"""
    #
    #     # state2 should be larger than state1
    #     assert state2 > state1, "State2 index should be larger than state1 index"
    #     assert self.T != None, "Set the temperature first"
    #
    #     decayList = []
    #
    #     # loop over all the external fluxes
    #     for i1 in range(len(self.HamilEigVecList)):
    #         # list of effective decay rate for each capacitor
    #         dieDecayList = [];
    #         qpDecayList = [];
    #
    #         ketState1 = self.HamilEigVecList[i1][state1]
    #         ketState2 = self.HamilEigVecList[i1][state2]
    #
    #         omegaState1 = self.HamilEigVal[state1, i1]
    #         omegaState2 = self.HamilEigVal[state2, i1]
    #
    #         omega_q = omegaState2 - omegaState1
    #
    #         # the vector that holds the expectation values of charge operators
    #         q_ge = np.array([(ketState1.dag() * Qtilde * ketState2)[0, 0] for Qtilde in self.chargeOpList])
    #
    #         # the decayVec is the vector which makes calculation easier and faster more
    #         # explanation in Qcircuit notes
    #         decayVec = self.cInv @ self.R @ q_ge
    #
    #         if (mode == 'dielectric' or mode == 'all'):
    #
    #             k_B = 1.38e-23
    #             # effect of tempreture
    #
    #             # prevent the exponential over flow(exp(709) is biggest number that numpy can calculate)
    #             if (hbar * (omega_q) / (k_B * self.T) > 709):
    #                 nbar = 0
    #             else:
    #                 nbar = 1 / (np.exp(hbar * (omega_q) / (k_B * self.T)) - 1)
    #
    #             for i, c_x in enumerate(self.C):
    #
    #                 # extracting the nodes
    #                 node1, node2 = self.graph[i]
    #
    #                 # check if the node is connected to ground
    #                 if (node1 == 0):
    #                     dieDecayList.append(
    #                         2 * c_x * self.dieTanList[i] * np.abs(decayVec[node2 - 1]) ** 2 * (2 * nbar + 1))
    #                 elif (node2 == 0):
    #                     dieDecayList.append(
    #                         2 * c_x * self.dieTanList[i] * np.abs(decayVec[node1 - 1]) ** 2 * (2 * nbar + 1))
    #
    #                 else:
    #                     dieDecayList.append(2 * c_x * self.dieTanList[i] *
    #                                         np.abs(decayVec[node1 - 1] - decayVec[node2 - 1]) ** 2 * (2 * nbar + 1))
    #
    #         if (mode == 'quasiparticles' or mode == 'all'):
    #             for i in range(len(self.JJEj)):
    #                 """https://www.researchgate.net/figure/SIS-parameters-for-cold-
    #                 and-room-temperature-evaporations-Number-of-samples-measured_tbl1_1896356
    #                 """
    #                 Delta = 250 * 1e-6 * 1.6e-19;
    #
    #                 # opExpt = (ketState1.dag()*(self.qpPrevList[i].dag() - self.qpPrevList[i])/(2j)*ketState2)[0,0]
    #                 opExpt = (ketState1.dag() * self.qpSinList[i1][i] * ketState2)[0, 0] + 1e-23
    #                 S = self.x_qpList[self.JJIndex[i]] * 8 * hbar * self.JJEj[i] / np.pi / hbar * np.sqrt(
    #                     2 * Delta / hbar / omega_q)
    #                 qpDecayList.append(np.abs(opExpt) ** 2 * S)
    #
    #         decayList.append(np.sum(dieDecayList) + np.sum(qpDecayList))
    #
    #     return np.array(decayList).real
    #
    # def setDrive(self, node):
    #     # function that builds the Hamiltonian related to drive the special node of the circuit.
    #     # The varibales are the same as the notation that I used in the notes.
    #
    #     # charge drive vector
    #     qd = np.zeros((self.n, 1));
    #     qd[node - 1] = 1
    #
    #     W_total = 0;
    #     weights = (qd.T @ self.cInv @ self.R)[0, :]
    #
    #     # calculate the W
    #     for i in range(self.n):
    #         W = 0
    #         for j in range(self.n):
    #             if (j == 0 and j == i):
    #                 if (self.omega[j] == 0):
    #                     Q = 2 * e / hbar * q.charge((self.m[j] - 1) / 2)
    #                     W = Q
    #                 else:
    #                     disMinCreat = q.destroy(self.m[j]) - q.create(self.m[j]);
    #                     coef = -1j * np.sqrt(1 / 2 * np.sqrt(self.lRotated[j, j] / self.cInvRotated[j, j]) / hbar)
    #                     W = coef * disMinCreat;
    #
    #             elif (j == 0 and j != i):
    #                 I = q.qeye(self.m[j])
    #                 W = I
    #
    #             if (j != 0 and j == i):
    #                 if (self.omega[j] == 0):
    #                     Q = 2 * e / hbar * q.charge((self.m[j] - 1) / 2)
    #                     W = q.tensor(W, Q)
    #                 else:
    #                     disMinCreat = q.destroy(self.m[j]) - q.create(self.m[j]);
    #                     coef = -1j * np.sqrt(1 / 2 * np.sqrt(self.lRotated[j, j] / self.cInvRotated[j, j]) / hbar)
    #                     W = q.tensor(W, coef * disMinCreat);
    #
    #             elif (j != 0 and j != i):
    #                 I = q.qeye(self.m[j])
    #                 W = q.tensor(w, I)
    #
    #         W_total = weights[i] * W
    #
    #     return W_total
    #
    # def getPotentialNode(self, node, phi, phiExt):
    #     # This function gives the potential related to speicific node as a function
    #     # of phase.(I'm assuming that the phase related to other nodes is zero)
    #
    #     potential = 1 / 2 * self.lRotated[node - 1, node - 1] * (phi * Phi0) ** 2 + 0j
    #
    #     # for i, EJ in enumerate(self.JJEj):
    #     #     potential -= hbar * EJ * np.cos(2*np.pi*self.JJEqRotated[i,node-1]*phi)
    #
    #     JJprevList = []
    #
    #     for i, EJ in enumerate(self.JJEj):
    #         JJprevList.append(np.exp(1j * 2 * np.pi * self.JJEqRotated[i, node - 1] * phi))
    #
    #         # List of individual JJ Hamiltonian
    #     potenList = [];
    #
    #     for i in range(len(self.JJEj)):
    #
    #         # add external excitation
    #
    #         # Parallel JJ case
    #         if (isinstance(self.JJEj[i], list)):
    #             JJEjIn, JJEjOut = self.JJEj[i]
    #             JJExcIn, JJExcOut = self.JJExcite[i]
    #             phiIn = 0;
    #             PhiOut = 0;
    #             if (JJExcIn):
    #                 for ind in JJExcIn:
    #                     phiIn = phiExt[ind];
    #             phiOut = phiIn + phiExt[JJExcOut];
    #
    #             H = -hbar * np.exp(1j * phiIn)
    #
    #             * JJEjIn / 2 * JJprevList[i] - hbar * np.exp(1j * phiOut) * JJEjOut / 2 * \
    #                 JJprevList[i]
    #             H = H + np.conj(H)
    #
    #         # single JJ case
    #         else:
    #             phi = 0
    #             # fluxoniumSituation
    #             if (len(self.graph) * len(self.L) == 1):
    #                 self.JJExcite = [[0]]
    #
    #             if (self.JJExcite[i]):
    #                 for ind in self.JJExcite[i]:
    #                     phi += phiExt[ind];
    #
    #             H = -hbar * np.exp(1j * phi) * self.JJEj[i] / 2 * JJprevList[i];
    #             H = H + np.conj(H)
    #
    #         potenList.append(H);
    #
    #     potential += sum(potenList)
    #
    #     return potential
