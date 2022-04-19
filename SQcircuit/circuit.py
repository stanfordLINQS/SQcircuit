# Libraries:
from SQcircuit.elements import *

import SQcircuit.physParam as phPar
import numpy as np
import qutip as q

import scipy.special
import scipy.sparse
from scipy.linalg import sqrtm, block_diag


class Circuit:
    """
    Class that contains the circuit properties and uses the theory discussed in the original
    paper of the SQcircuit to calculate:

        * Eigenvalues and eigenvectors
        * Phase coordinate representation of eigenvectors
        * Coupling operators
        * Matrix elements
        * Decoherence rates
        * Robustness analysis

    Parameters
    ----------
        circuitElements: dict
            A dictionary that contains the circuit's elements at each branch of the circuit.
        random: bool
            If `True`, each element of the circuit is a random number due to fabrication error. This
            is necessary for robustness analysis.
        fluxDist: str
            Provide the method of distributing the external fluxes. If `fluxDist` is `None`( the default value)
            , the SQcircuit assign the external fluxes based on the capacitor of each inductive element.
            If `fluxDist` is `"inductor"` SQcircuit finds the external flux distribution by assuming the
            capacitor of the inductors are much smaller than the junction capacitors, If `fluxDist` is `"junction"`
            it is the other way around.
    """

    # external charges of the circuit
    extCharge = {}
    # list of charge operators( transformed operators) (self.n)
    chargeOpList = []
    # list of flux operators(transformed operators) (self.n)
    fluxOpList = []
    # cross multiplication of charge operators as list
    chargeByChargeList = []
    # list of number operators (self.n)
    numOpList = []
    # LC part of the Hamiltonian
    HLC = q.Qobj()
    # List of exponential part of the Josephson Junction cosine
    HJJExpList = []
    # List of square root of exponential part of
    HJJExpRootList = []
    # sin(phi/2) operator related to each JJ for quasi-particle Loss
    qpSinList = []

    # eigenvalues of the circuit
    hamilEigVal = []
    # eigenvectors of the circuit
    hamilEigVec = []

    # temperature of the circuit
    T = 0.015
    # low-frequency cut off
    omegaLow = 2 * np.pi
    # high-frequency cut off
    omegaHigh = 2 * np.pi * 3 * 1e9
    # experiment time
    tExp = 10e-6

    def __init__(self, circuitElements: dict, random: bool = False, fluxDist: str = None):

        # circuit inductive loops
        self.loops = []

        # external charges of the circuit
        self.extCharge = {}

        # loop distribution over inductive elements.
        self.K2 = None

        self.circuitElements = circuitElements

        self.random = random

        self.fluxDist = fluxDist

        # number of nodes
        self.n = max(max(self.circuitElements))

        # number of branches that contain JJ without parallel inductor.
        self.countJJnoInd = 0

        # get the capacitance matrix, inductance matrix, and w matrix
        self.C, self.L, self.W = self.loopLCW()

        # the inverse of transformation of coordinates for charge operators
        self.R = np.zeros((self.n, self.n))

        # the inverse of transformation of coordinates for flux operators
        self.S = np.zeros((self.n, self.n))

        # S and R matrix of first, second, and third transformation
        self.R1 = np.zeros((self.n, self.n))
        self.S1 = np.zeros((self.n, self.n))
        self.R2 = np.zeros((self.n, self.n))
        self.S2 = np.zeros((self.n, self.n))
        self.R3 = np.zeros((self.n, self.n))
        self.S3 = np.zeros((self.n, self.n))

        # diagonalized sudo-inductance matrix
        self.lDiag = np.zeros((self.n, self.n))
        # transformed inverse capacitance matrix
        self.cInvDiag = np.zeros((self.n, self.n))

        # natural angular frequencies of the circuit(zero for modes in charge basis)
        self.omega = np.zeros(self.n)

        # transformed w matrix
        self.wTrans = np.zeros_like(self.W)

        # transform the Hamiltonian of the circuit
        self.transformHamil()

        # truncation numbers for each mode
        self.m = []
        # squeezed truncation numbers( eliminating the modes with truncation number equals 1)
        self.ms = []

    def __getstate__(self):
        attrs = self.__dict__
        typeAttrs = type(self).__dict__
        selfDict = {k: attrs[k] for k in attrs if k not in typeAttrs}
        return selfDict

    def __setstate__(self, state):
        self.__dict__ = state

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

    @staticmethod
    def _independentRows(A):
        """use Gram–Schmidt to find the linear independent rows of matrix A
        """
        # normalize the row of matrix A
        A_norm = A / np.linalg.norm(A, axis=1).reshape(A.shape[0], 1)

        basis = []
        indList = []

        for i, a in enumerate(A_norm):
            aPrime = a - sum([np.dot(a, e) * e for e in basis])
            if (np.abs(aPrime) > 1e-7).any():
                indList.append(i)
                basis.append(aPrime / np.linalg.norm(aPrime))

        return indList, basis

    def addLoop(self, loop):
        """
        Add loop to the circuit loops.
        """
        if loop not in self.loops:
            loop.reset()
            self.loops.append(loop)

    def loopLCW(self):
        """
        calculate the capacitance matrix, inductance matrix, w matrix, and the flux distribution over
        inductive elements.
        outputs:
            -- cMat: capacitance matrix (self.n,self.n)
            -- lMat: inductance matrix (self.n,self.n)
            -- wMat:  W matrix(linear combination of the flux node operators in the JJ cosine (n_J,self.n)
        """

        cMat = np.zeros((self.n, self.n))
        lMat = np.zeros((self.n, self.n))
        wMat = []

        # count of inductive elements
        count = 0

        # number of branches that contain JJ without parallel inductor.
        countJJnoInd = 0

        # K1 is a matrix that transfer node coordinates to edge phase drop for inductive elements
        K1 = []
        # capacitor at each inductive elements
        cEd = []

        for edge in self.circuitElements.keys():
            # i1 and i2 are the nodes of the edge
            i1, i2 = edge

            w = [0] * (self.n + 1)

            if i1 == 0 or i2 == 0:
                w[i1 + i2] += 1
            else:
                w[i1] += 1
                w[i2] -= 1

            # elements of the edge
            edgeElements = self.circuitElements[edge]

            # list of capacitors of the edge.
            capList = []
            # list of inductors of the edge
            indList = []
            # list of Josephson Junction of the edge.
            JJList = []

            for el in edgeElements:

                if isinstance(el, Capacitor):
                    capList.append(el)

                elif isinstance(el, Inductor):
                    indList.append(el)
                    # capacitor of inductor
                    capList.append(el.cap)
                    loops = el.loops
                    for loop in loops:
                        self.addLoop(loop)
                        loop.addIndex(count)
                        loop.addK1(w[1:])

                    K1.append(w[1:])
                    if self.fluxDist is None:
                        cEd.append(el.cap.value())
                    elif self.fluxDist == "junction" or self.fluxDist == "Junction":
                        cEd.append(Capacitor(1e20).value())
                    elif self.fluxDist == "inductor" or self.fluxDist == "Inductor":
                        cEd.append(Capacitor().value())

                    count += 1

                elif isinstance(el, Junction):
                    JJList.append(el)
                    # capacitor of JJ
                    capList.append(el.cap)
                    loops = el.loops
                    for loop in loops:
                        self.addLoop(loop)
                        loop.addIndex(count)
                        loop.addK1(w[1:])

                    K1.append(w[1:])
                    if self.fluxDist is None:
                        cEd.append(el.cap.value())
                    elif self.fluxDist == "junction" or self.fluxDist == "Junction":
                        cEd.append(Capacitor().value())
                    elif self.fluxDist == "inductor" or self.fluxDist == "Inductor":
                        cEd.append(Capacitor(1e20).value())

                    count += 1

            if len(indList) == 0 and len(JJList) != 0:
                countJJnoInd += 1

            # summation of the capacitor values.
            cap = sum(list(map(lambda c: c.value(self.random), capList)))

            # summation of the one over inductor values.
            x = np.sum(1 / np.array(list(map(lambda l: l.value(self.random), indList))))

            if i1 != 0 and i2 == 0:
                cMat[i1 - 1, i1 - 1] += cap
                lMat[i1 - 1, i1 - 1] += x
            elif i1 == 0 and i2 != 0:
                cMat[i2 - 1, i2 - 1] += cap
                lMat[i2 - 1, i2 - 1] += x
            else:
                cMat[i1 - 1, i2 - 1] = - cap
                cMat[i2 - 1, i1 - 1] = - cap
                cMat[i1 - 1, i1 - 1] += cap
                cMat[i2 - 1, i2 - 1] += cap
                lMat[i1 - 1, i2 - 1] = -x
                lMat[i2 - 1, i1 - 1] = -x
                lMat[i1 - 1, i1 - 1] += x
                lMat[i2 - 1, i2 - 1] += x

            if len(JJList) != 0:
                wMat.append(w[1:])

        wMat = np.array(wMat)

        K1 = np.array(K1)
        a = np.zeros_like(K1)
        select = np.sum(K1 != a, axis=0) != 0
        # eliminate the zero columns
        K1 = K1[:, select]
        if K1.shape[0] == K1.shape[1]:
            K1 = K1[:, 0:-1]

        X = K1.T @ np.diag(cEd)
        for loop in self.loops:
            p = np.zeros((1, count))
            p[0, loop.indices] = loop.getP()
            X = np.concatenate((X, p), axis=0)
        # number of inductive loops of the circuit
        numLoop = len(self.loops)

        if numLoop != 0:
            Y = np.concatenate((np.zeros((count - numLoop, numLoop)), np.eye(numLoop)), axis=0)
            self.K2 = np.linalg.inv(X) @ Y
            self.K2 = np.around(self.K2, 5)

        self.countJJnoInd = countJJnoInd

        return cMat, lMat, wMat

    def transform1(self):
        """
        First transformation of the coordinates that simultaneously diagonalizes
        the capacitance and inductance matrices.

        output:
            --  lDiag: diagonalized sudo-inductance matrix (self.n,self.n)
            --  cInvDiag: diagonalized inverse of capacitance matrix (self.n,self.n)
            --  R1: transformation of charge operators (self.n,self.n)
            --  S1: transformation of flux operators (self.n,self.n)
        """

        # cMat = self.getMatC()
        cMat = self.C
        # lMat = self.getMatL()
        lMat = self.L
        cMatInv = np.linalg.inv(cMat)

        cMatRoot = sqrtm(cMat)
        cMatRootInv = np.linalg.inv(cMatRoot)
        lMatRoot = sqrtm(lMat)

        V, D, U = np.linalg.svd(lMatRoot @ cMatRootInv)

        # the case that there is not any inductor in the circuit
        if np.max(D) == 0:
            D = np.diag(np.eye(self.n))
            singLoc = list(range(0, self.n))
        else:
            # find the number of singularity in the circuit
            lEig, _ = np.linalg.eig(lMat)
            numSing = len(lEig[lEig / np.max(lEig) < 1e-11])
            singLoc = list(range(self.n - numSing, self.n))
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
        # wTrans1 = self.getMatW() @ S1
        wTrans1 = self.W @ S1
        wQ = wTrans1[:, omega == 0]

        # wQ[np.abs(wQ) < 1e-2] = 0
        # a = np.zeros_like(wQ)
        # select = np.sum(wQ != a, axis=0) != 0
        # # eliminate the zero columns
        # wQ = wQ[:, select]

        # number of operators represented in charge bases
        nq = wQ.shape[1]

        # if we need to represent an operator in charge basis
        if nq != 0 and self.countJJnoInd != 0:

            # normalizing the wQ vectors(each row is a vector)
            # wQ_norm = wQ / np.linalg.norm(wQ, axis=1).reshape(wQ.shape[0], 1)

            # list of indices of w vectors that are independent
            indList = []

            X = []
            # use Gram–Schmidt to find the linear independent rows of normalized wQ (wQ_norm)
            basis = []
            while len(basis) != nq:
                if len(basis) == 0:
                    indList, basis = self._independentRows(wQ)
                else:
                    # to complete the basis
                    X = list(np.random.randn(nq - len(basis), nq))
                    basisComplete = np.array(basis + X)
                    _, basis = self._independentRows(basisComplete)

                # for i, w in enumerate(wQ_norm):
                #     wPrime = w - sum([np.dot(w, e) * e for e in basis])
                #     if (np.abs(wPrime) > 1e-7).any():
                #         indList.append(i)
                #         basis.append(wPrime / np.linalg.norm(wPrime))

            # the second S and R matrix are:
            F = np.array(list(wQ[indList, :]) + X)
            S2 = block_diag(np.eye(self.n - nq), np.linalg.inv(F))

            # S2 = block_diag(np.eye(self.n - nq), np.linalg.inv(wQ[indList, :]))
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
                alpha = np.abs(2 * np.pi / phPar.Phi0 * np.sqrt(phPar.hbar / 2 * np.sqrt(
                    self.cInvDiag[j, j] / self.lDiag[j, j])) * self.wTrans[:, j])

                self.wTrans[:, j][alpha < 1e-11] = 0
                if np.max(alpha) > 1e-11:
                    # find the coefficient in wTrans for j-th mode that has maximum alpha
                    s = np.abs(self.wTrans[np.argmax(alpha), j])
                    # scale that mode with s
                    self.wTrans[:, j] = self.wTrans[:, j] / s
                    S3[j, j] = 1 / s
                    for i in range(self.n):
                        if i == j:
                            self.cInvDiag[i, j] *= s ** 2
                            self.lDiag[i, j] /= s ** 2
                        else:
                            self.cInvDiag[i, j] *= s
                            self.lDiag[i, j] /= s
                else:
                    # scale the uncoupled mode
                    S = np.abs(self.S1 @ self.S2)

                    s = np.max(S[:, j])

                    S3[j, j] = 1 / s
                    for i in range(self.n):
                        if i == j:
                            self.cInvDiag[i, j] *= s ** 2
                            self.lDiag[i, j] /= s ** 2
                        else:
                            self.cInvDiag[i, j] *= s
                            self.lDiag[i, j] /= s

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

        # set the external charge for each charge mode.
        self.extCharge = {i: Charge() for i in range(self.n) if self.omega[i] == 0}

        # get the second transformation:
        self.S2, self.R2 = self.transform2(self.omega, self.S1)

        # apply the second transformation on self.cInvDiag
        self.cInvDiag = self.R2.T @ self.cInvDiag @ self.R2

        # get the transformed W matrix
        self.wTrans = self.W @ self.S1 @ self.S2
        if self.countJJnoInd == 0:
            self.wTrans[:, self.omega == 0] = 0
        # wQ = self.wTrans[:, self.omega == 0]
        # wQ[np.abs(wQ) < 0.98] = 0
        # self.wTrans[:, self.omega == 0] = wQ

        # scaling the modes
        self.S3, self.R3 = self.transform3()

        # The final transformations are:
        self.S = self.S1 @ self.S2 @ self.S3
        self.R = self.R1 @ self.R2 @ self.R3

    def description(self):
        """
        Print out a listing of the modes, whether they are harmonic or
        charge modes, and the frequency for each harmonic mode. Moreover, it shows the
        prefactors in the Josephson junction part of the Hamiltonian :math:`w_k^T`
        """

        for i in range(self.n):
            if self.omega[i] != 0:
                print("mode_{}: \tharmonic\tfreq={}".format(i + 1, self.omega[i] / (2 * np.pi * phPar.freq)))
            else:
                print("mode_{}: \tcharge".format(i + 1))

        for i in range(self.wTrans.shape[0]):
            print("w{}: \t{}".format(i + 1, self.wTrans[i, :]))

    def truncationNumbers(self, truncNum: list):
        """Set the truncation numbers for each mode.

        Parameters
        ----------
            truncNum: list
                A list that contains the truncation numbers for each mode.
        """

        error1 = "The input must be be a python list"
        assert isinstance(truncNum, list), error1
        error2 = "The number of modes(length of the input) must be equal to the number of nodes"
        assert len(truncNum) == self.n, error2
        self.m = truncNum

        # squeeze the mode with truncation number equal to 1.
        self.ms = list(filter(lambda x: x != 1, self.m))

        self.chargeOpList, self.numOpList, self.chargeByChargeList, self.fluxOpList = self.buildOpMemory(
            self.lDiag, self.cInvDiag, self.omega)

        self.HLC = self.getLCHamil(self.cInvDiag, self.omega, self.chargeByChargeList, self.numOpList)

        self.HJJExpList, self.HJJExpRootList = self.getHJJExp(self.cInvDiag, self.lDiag, self.omega, self.wTrans)

    def chargeOffset(self, mode: int, ng: float):
        """set the charge offset for each charge mode.

        Parameters
        ----------
            mode: int
                An integer that specifies the charge mode. To see, which mode is a charge mode, one
                can use `description()` method.
            ng: float
                The charge offset.
        """
        assert isinstance(mode, int), "Mode number should be an integer"
        assert mode-1 in self.extCharge, "The specified mode is not a charge mode."
        if len(self.m) == 0:
            self.extCharge[mode-1].setOffset(ng)
        else:
            self.extCharge[mode-1].setOffset(ng)
            self.chargeOpList, self.numOpList, self.chargeByChargeList, self.fluxOpList = self.buildOpMemory(
                self.lDiag, self.cInvDiag, self.omega)

            self.HLC = self.getLCHamil(self.cInvDiag, self.omega, self.chargeByChargeList, self.numOpList)

    def chargeNoise(self, mode: int, A: float):
        """set the charge noise for each charge mode.

        Parameters
        ----------
            mode: int
                An integer that specifies the charge mode. To see, which mode is a charge mode, one
                can use `description()` method.
            A: float
                The charge noise.
        """
        assert isinstance(mode, int), "Mode number should be an integer"
        assert mode-1 in self.extCharge, "The specified mode is not a charge mode."

        self.extCharge[mode-1].setNoise(A)

    def buildOpMemory(self, lDiag: np.array, cInvDiag: np.array, omega: np.array):
        """
        build the charge operators, number operators, and cross multiplication of
        charge operators.
        inputs:
            -- lDiag: diagonalized inductance matrix (self.n,self.n)
            -- cInvDiag: diagonalized inverse of capacitance matrix (self.n,self.n)
            -- omega: natural frequencies of the circuit (self.n)
        outputs:
            -- chargeOpList : list of charge operators (self.n)
            -- fluxOpList: list of flux operators (self.n)
            -- chargeByChargeList : cross multiplication of charge operators as list
            -- numOpList : list of number operators (self.n)
        """

        chargeOpList = []
        fluxOpList = []
        numOpList = []
        chargeByChargeList = []

        # list of charge operators in their own mode basis
        # (tensor product of other modes are not applied yet!)
        QList = []
        for i in range(self.n):
            if omega[i] == 0:
                Q0 = (2 * phPar.e / np.sqrt(phPar.hbar)) * q.charge((self.m[i] - 1) / 2) - \
                     (2 * phPar.e / np.sqrt(phPar.hbar)) * self.extCharge[i].value()
            else:
                coef = -1j * np.sqrt(1 / 2 * np.sqrt(lDiag[i, i] / cInvDiag[i, i]))
                Q0 = coef * (q.destroy(self.m[i]) - q.create(self.m[i]))
            QList.append(Q0)

        fluxList = []
        # list of flux operators in their own mode basis
        # (tensor product of other modes are not applied yet!)
        for i in range(self.n):
            if omega[i] == 0:
                flux0 = q.qeye(self.m[i])
            else:
                coef = np.sqrt(1 / 2 * np.sqrt(cInvDiag[i, i] / lDiag[i, i]))
                flux0 = coef * (q.destroy(self.m[i]) + q.create(self.m[i]))
            fluxList.append(flux0)

        # list of number operators in their own mode basis
        # (tensor product of other modes are not applied yet!)
        nList = []
        for i in range(self.n):
            if omega[i] == 0:
                num0 = q.charge((self.m[i] - 1) / 2)
            else:
                num0 = q.num(self.m[i])
            nList.append(num0)

        for i in range(self.n):
            chargeRowList = []
            num = q.Qobj()
            Q = q.Qobj()
            flux = q.Qobj()
            for j in range(self.n):
                # find the appropriate charge and number operator for first mode
                if j == 0 and i == 0:
                    Q2 = QList[j] * QList[j]
                    Q = QList[j]
                    num = nList[j]
                    flux = fluxList[j]

                    # Tensor product the charge with I for other modes
                    for k in range(self.n - 1):
                        Q2 = q.tensor(Q2, q.qeye(self.m[k + 1]))
                    chargeRowList.append(Q2)

                elif j == 0 and i != 0:
                    I = q.qeye(self.m[j])
                    Q = I
                    num = I
                    flux = I

                # find the rest of the modes
                elif j != 0 and j < i:
                    I = q.qeye(self.m[j])
                    Q = q.tensor(Q, I)
                    num = q.tensor(num, I)
                    flux = q.tensor(flux, I)

                elif j != 0 and j == i:
                    Q2 = q.tensor(Q, QList[j] * QList[j])
                    Q = q.tensor(Q, QList[j])
                    num = q.tensor(num, nList[j])
                    flux = q.tensor(flux, fluxList[j])

                    # Tensor product the charge with I for other modes
                    for k in range(self.n - j - 1):
                        Q2 = q.tensor(Q2, q.qeye(self.m[k + j + 1]))
                    chargeRowList.append(Q2)

                elif j > i:
                    QQ = q.tensor(Q, QList[j])

                    # Tensor product the QQ with I for other modes
                    for k in range(self.n - j - 1):
                        QQ = q.tensor(QQ, q.qeye(self.m[k + j + 1]))
                    chargeRowList.append(QQ)

                    I = q.qeye(self.m[j])
                    Q = q.tensor(Q, I)
                    num = q.tensor(num, I)
                    flux = q.tensor(flux, I)

            chargeOpList.append(Q)
            numOpList.append(num)
            chargeByChargeList.append(chargeRowList)
            fluxOpList.append(flux)

        return chargeOpList, numOpList, chargeByChargeList, fluxOpList

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

    def getHJJExp(self, cInvDiag: np.array, lDiag: np.array, omega: np.array, wTrans: np.array):
        """
        Each cosine potential of the Josephson Junction can be written as summation of two
        exponential terms,cos(x)=(exp(ix)+exp(-ix))/2. This function returns the quantum
        operators for only one exponential term.
        inputs:
            -- lDiag: diagonalized inductance matrix (self.n,self.n)
            -- cInvDiag: diagonalized inverse of capacitance matrix (self.n,self.n)
            -- omega: natural frequencies of the circuit (self.n)
            -- wTrans: transformed W matrix (nJ,self.n)
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
            for j in range(self.n):
                if j == 0 and omega[j] == 0:
                    if wTrans[i, j] == 0:
                        I = q.qeye(self.m[j])
                        H = I
                        H2 = I
                    elif wTrans[i, j] > 0:
                        d = self.chargeDisp(self.m[j])
                        I = q.qeye(self.m[j])
                        H = d
                        # not correct just to avoid error:
                        H2 = I
                    else:
                        d = self.chargeDisp(self.m[j])
                        I = q.qeye(self.m[j])
                        H = d.dag()
                        # not correct just to avoid error:
                        H2 = I

                elif j == 0 and omega[j] != 0:
                    alpha = 2 * np.pi / phPar.Phi0 * 1j * np.sqrt(
                        phPar.hbar / 2 * np.sqrt(cInvDiag[j, j] / lDiag[j, j])) * wTrans[i, j]
                    H = q.displace(self.m[j], alpha)
                    H2 = q.displace(self.m[j], alpha / 2)

                if j != 0 and omega[j] == 0:
                    if wTrans[i, j] == 0:
                        I = q.qeye(self.m[j])
                        H = q.tensor(H, I)
                        H2 = q.tensor(H2, I)
                    elif wTrans[i, j] > 0:
                        I = q.qeye(self.m[j])
                        d = self.chargeDisp(self.m[j])
                        H = q.tensor(H, d)
                        H2 = q.tensor(H2, I)
                    else:
                        I = q.qeye(self.m[j])
                        d = self.chargeDisp(self.m[j])
                        H = q.tensor(H, d.dag())
                        H2 = q.tensor(H2, I)

                elif j != 0 and omega[j] != 0:
                    alpha = 2 * np.pi / phPar.Phi0 * 1j * np.sqrt(
                        phPar.hbar / 2 * np.sqrt(cInvDiag[j, j] / lDiag[j, j])) * wTrans[i, j]
                    H = q.tensor(H, q.displace(self.m[j], alpha))
                    H2 = q.tensor(H2, q.displace(self.m[j], alpha / 2))

            HJJExpList.append(H)
            HJJExpRootList.append(H2)

        return HJJExpList, HJJExpRootList

    def indHamil(self, HJJExpList: list, HJJExpRootList: list):

        countInd = 0
        countJJ = 0
        JJFlag = False
        H = q.Qobj()
        self.junctionHamil = {'cos': {}, 'sin': {}, 'sinHalf': {}}
        self.inductorHamil = {}

        for edge in self.circuitElements.keys():

            # elements of the edge
            edgeElements = self.circuitElements[edge]

            for el in edgeElements:

                if isinstance(el, Inductor):

                    phi = 0
                    for i, loop in enumerate(self.loops):
                        phi += loop.value(self.random) * self.K2[countInd, i]

                    # summation of the 1 over inductor values.
                    x = 1 / el.value(self.random)
                    O = self.couplingOperator("inductive", edge)
                    O.dims = [self.m, self.m]
                    H += x * phi * (phPar.Phi0 / 2 / np.pi) * O / np.sqrt(phPar.hbar)

                    # save the operators for loss calculation
                    self.inductorHamil[(countInd, el)] = np.sqrt(x) * O

                    countInd += 1

                if isinstance(el, Junction):

                    phi = 0
                    for i, loop in enumerate(self.loops):
                        phi += loop.value(self.random) * self.K2[countInd, i]

                    EJ = el.value(self.random)
                    HJ_exp = np.exp(1j * phi) * EJ / 2 * HJJExpList[countJJ]
                    HJ_expRoot = np.exp(1j * phi / 2) * np.sqrt(EJ) / 2 * HJJExpRootList[countJJ]
                    HJ = HJ_exp + HJ_exp.dag()
                    H -= HJ

                    # save the operators for loss calculations.
                    self.junctionHamil['cos'][el] = HJ
                    self.junctionHamil['sin'][(countInd, el)] = (HJ_exp - HJ_exp.dag()) / 1j
                    self.junctionHamil['sinHalf'][el] = (HJ_expRoot - HJ_expRoot.dag()) / 1j

                    countInd += 1
                    JJFlag = True

            if JJFlag:
                countJJ += 1
                JJFlag = False

        return H

    def diag(self, numEig: int):
        """
        Diagonalize the Hamiltonian of the circuit and return the eigenrfequencies and eigenvectors of the circuit up
        to specified number of eigenvalues.

        Parameters
        ----------
            numEig: int
                The number of eigenvalues to output. The lower `numEig`, the faster `SQcircuit` finds
                the eigenvalues.
        """
        assert len(self.m) != 0, "Please specify the truncation number for each mode."
        assert isinstance(numEig, int), "The numEig( number of eigenvalues) should be an integer."

        H = self.hamiltonian()

        # get the data out of qutip variable and use sparse scipy eigen solver which is faster than
        # non-sparse eigen solver
        eigenValues, eigenVectors = scipy.sparse.linalg.eigs(H.data, numEig, which='SR')
        # the output of eigen solver is not sorted
        eigenValuesSorted = np.sort(eigenValues.real)
        sortArg = np.argsort(eigenValues)
        eigenVectorsSorted = [q.Qobj(eigenVectors[:, ind], dims=[self.ms, len(self.ms) * [1]])
                              for ind in sortArg]

        # store the eigenvalues and eigenvectors of the circuit Hamiltonian
        self.hamilEigVal = eigenValuesSorted
        self.hamilEigVec = eigenVectorsSorted

        return eigenValuesSorted.real / (2 * np.pi * phPar.freq), eigenVectorsSorted

    ###############################################
    # Methods that calculate circuit properties
    ###############################################

    def coordinateTransformation(self, opType: str):
        """
        Return the transformation of the coordinates for each type of operators, either charge or flux.

        Parameters
        ----------
            opType: str
                The type of the operators that can be either `"charge"` or `"flux"`.
        """
        if opType == "charge" or opType == "Charge":
            return np.linalg.inv(self.R)
        elif opType == "flux" or opType == "Flux":
            return np.linalg.inv(self.S)
        else:
            raise ValueError(" The input must be either \"charge\" or \"flux\".")

    def hamiltonian(self):
        """
        Returns the transformed hamiltonian of the circuit as QuTiP object.
        """
        assert len(self.m) != 0, "Please specify the truncation number for each mode."

        Hind = self.indHamil(self.HJJExpList, self.HJJExpRootList)

        H = Hind + self.HLC

        return H

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

    def eigPhaseCoordinate(self, eigInd: int, grid: list):
        """
        Return the phase coordinate representations of the eigenvectors.

        Parameters
        ----------
            eigInd: int
                The eigenvector index. For example, we set it to 0 for the ground state and 1
                for the first excited state.
            grid: list
                A list that contains the range of values of phase φ for which we want to evaluate the
                wavefunction.
        """

        assert isinstance(eigInd, int), "The eigInd( eigen index) should be an integer."

        phiList = [*np.meshgrid(*grid, indexing='ij')]

        # The total dimension of the circuit Hilbert Space
        netDimension = np.prod(self.m)

        state = 0

        for i in range(netDimension):

            # decomposes the tensor product space index (i) to each mode indices as a list
            indList = self.tensorToModes(i)

            term = self.hamilEigVec[eigInd][i][0, 0]

            for mode in range(self.n):

                # mode number related to that node
                n = indList[mode]

                # For charge basis
                if self.omega[mode] == 0:
                    term *= 1 / np.sqrt(2 * np.pi) * np.exp(1j * phiList[mode] * n)
                # For harmonic basis
                else:
                    x0 = np.sqrt(phPar.hbar * np.sqrt(self.cInvDiag[mode, mode] / self.lDiag[mode, mode]))

                    coef = 1 / np.sqrt(np.sqrt(np.pi) * 2 ** n * scipy.special.factorial(n) * x0 / phPar.Phi0)

                    term *= coef * np.exp(-(phiList[mode] * phPar.Phi0 / x0) ** 2 / 2) * \
                            scipy.special.eval_hermite(n, phiList[mode] * phPar.Phi0 / x0)

            state += term

        state = np.squeeze(state)

        # transposing the first two modes
        if len(state.shape) > 1:
            indModes = list(range(len(state.shape)))
            indModes[0] = 1
            indModes[1] = 0
            state = state.transpose(*indModes)

        return state

    def couplingOperator(self, copType: str, nodes: tuple):
        """
        Return the capacitive or inductive coupling operator related to the specified nodes. The output has the
        QuTip object format.

        Parameters
        ----------
            copType: str
                Coupling type which is either `"capacitive"` or `"inductive"`.
            nodes: tuple
                A tuple of circuit nodes to which we want to couple.
        """
        error = "The coupling type must be either \"capacitive\" or \"inductive\""
        assert copType in ["capacitive", "inductive"], error
        assert isinstance(nodes, tuple) or isinstance(nodes, list), "Nodes must be either a list or a set."

        op = q.Qobj()

        node1 = nodes[0]
        node2 = nodes[1]

        # for the case that we have ground in the edge
        if 0 in nodes:
            node = node1 + node2
            if copType == "capacitive":
                # K = np.linalg.inv(self.getMatC()) @ self.R
                K = np.linalg.inv(self.C) @ self.R
                for i in range(self.n):
                    op += K[node - 1, i] * self.chargeOpList[i]
            if copType == "inductive":
                K = self.S
                for i in range(self.n):
                    op += K[node - 1, i] * self.fluxOpList[i]

        else:
            if copType == "capacitive":
                # K = np.linalg.inv(self.getMatC()) @ self.R
                K = np.linalg.inv(self.C) @ self.R
                for i in range(self.n):
                    op += (K[node2 - 1, i] - K[node1 - 1, i]) * self.chargeOpList[i]
            if copType == "inductive":
                K = self.S
                for i in range(self.n):
                    op += (K[node1 - 1, i] - K[node2 - 1, i]) * self.fluxOpList[i]

        # squeezing the dimension
        op.dims = [self.ms, self.ms]

        return op

    def matrixElements(self, copType: str, nodes: tuple, states: tuple):
        """
        Return the matrix element of two eigenstates for either capacitive or inductive coupling.

        Parameters
        ----------
            copType: str
                Coupling type which is either `"capacitive"` or `"inductive"`.
            nodes: tuple
                A tuple of circuit nodes to which we want to couple.
            states: tuple
                A tuple of indices of eigenstates for which we want to calculate the matrix element.
        """

        state1 = self.hamilEigVec[states[0]]
        state2 = self.hamilEigVec[states[1]]

        # get the coupling operator
        op = self.couplingOperator(copType, nodes)

        return (state1.dag() * op * state2).data[0, 0]

    def setTemperature(self, T: float):
        """
        Set the temperature of the circuit.

        Parameters
        ----------
            T: float
                The temperature in Kelvin
        """
        self.T = T

    def setLowFreq(self, value: float, unit: str):
        """
        Set the low-frequency cut-off.

        Parameters
        ----------
            value: The value of the frequency.
            unit: The unit of the input value in hertz unit that can be "THz", "GHz", "MHz",and ,etc.
        """
        self.omegaLow = 2 * np.pi * value * phPar.freqList[unit]

    def setHighFreq(self, value: float, unit: str):
        """
        Set the high-frequency cut-off.

        Parameters
        ----------
            value: The value of the frequency.
            unit: The unit of the input value in hertz unit that can be "THz", "GHz", "MHz",and ,etc.
        """
        self.omegaHigh = 2 * np.pi * value * phPar.freqList[unit]

    def setTexp(self, value: float, unit: str):
        """
        Set the measurement time.

        Parameters
        ----------
            value: The value of the measurement time.
            unit: The unit of the input value in time unit that can be "s", "ms", "us",and ,etc.
        """
        self.tExp = value * phPar.timeList[unit]

    def decRate(self, decType: str, states: tuple, total: bool = True):
        """ Return the decoherence rate in [1/s] between each two eigenstates for different types of
        depolarization and dephasing.

        Parameters
        ----------
            decType: str
                decoherence type that can be: `"capacitive"` for capacitive loss; `"inductive"` for inductive loss;
                `"quasiparticle"` for quasiparticle loss; `"charge"` for charge noise, `"flux"` for flux noise; and
                `"cc"` for critical current noise.
            states: tuple
                A tuple of indices of eigenstates for which we want to calculate the decoherence rate. For example,
                for `states=(0,1)`, we calculate the decoherence rate between the ground state and the first excited
                state.
            total:

        """

        omega1 = self.hamilEigVal[states[0]]
        omega2 = self.hamilEigVal[states[1]]

        state1 = self.hamilEigVec[states[0]]
        state2 = self.hamilEigVec[states[1]]

        omega = np.abs(omega2 - omega1)

        decay = 0

        # prevent the exponential overflow(exp(709) is the biggest number that numpy can calculate)
        if phPar.hbar * omega / (phPar.k_B * self.T) > 709:
            down = 2
            up = 0
        else:
            alpha = phPar.hbar * omega / (phPar.k_B * self.T)
            down = (1 + 1 / np.tanh(alpha / 2))
            up = down * np.exp(-alpha)

        # for temperature dependent loss
        if not total:
            if states[0] > states[1]:
                tempS = down
            else:
                tempS = up
        else:
            tempS = down + up

        if decType == "capacitive":

            for edge in self.circuitElements.keys():

                for el in self.circuitElements[edge]:
                    if isinstance(el, Capacitor):
                        cap = el
                    else:
                        cap = el.cap

                    if cap.Q:
                        decay += tempS * cap.value() / cap.Q(omega) * np.abs(self.matrixElements(
                            "capacitive", edge, states)) ** 2

        if decType == "inductive":

            for indx, el in self.inductorHamil:
                op = self.inductorHamil[(indx, el)]
                op.dims = [self.ms, self.ms]
                if el.Q:
                    decay += tempS / el.Q(omega, self.T) * np.abs((state1.dag() * op * state2).data[0, 0]) ** 2

        if decType == "quasiparticle":

            for el in self.junctionHamil['sinHalf']:
                op = self.junctionHamil['sinHalf'][el]
                op.dims = [self.ms, self.ms]

                # Delta = 0.00025 * 1.6e-19
                # Y = el.x_qp * 8 * phPar.hbar / np.pi / phPar.hbar * np.sqrt(2 * Delta / phPar.hbar / omega)

                decay += tempS * el.Y(omega, self.T) * omega \
                         * phPar.hbar * np.abs((state1.dag() * op * state2).data[0, 0]) ** 2
        elif decType == "charge":

            # first derivative of the Hamiltonian with respect to charge noise
            op = q.Qobj()
            for i in range(self.n):
                if self.omega[i] == 0:
                    for j in range(self.n):
                        op += self.cInvDiag[i, j] * self.chargeOpList[j] / np.sqrt(phPar.hbar)
                    op.dims = [self.ms, self.ms]
                    partialOmega = np.abs((state2.dag() * op * state2 - state1.dag() * op * state1).data[0, 0])
                    decay += partialOmega * (self.extCharge[i].A * 2 * phPar.e) \
                             * np.sqrt(2 * np.abs(np.log(self.omegaLow * self.tExp)))

        elif decType == "cc":
            for el in self.junctionHamil['cos']:
                op = self.junctionHamil['cos'][el]
                op.dims = [self.ms, self.ms]
                partialOmega = np.abs((state2.dag() * op * state2 - state1.dag() * op * state1).data[0, 0])
                decay += partialOmega * el.A * np.sqrt(2 * np.abs(np.log(self.omegaLow * self.tExp)))

        elif decType == "flux":

            for indx, el in self.inductorHamil:
                op = self.inductorHamil[(indx, el)]
                op.dims = [self.ms, self.ms]
                partialOmega = np.abs(
                    (state2.dag() * op * state2 - state1.dag() * op * state1).data[0, 0]) / np.sqrt(el.value())

                A = 0
                for i, loop in enumerate(self.loops):
                    A += loop.A * self.K2[indx, i] * phPar.Phi0

                decay += partialOmega * A * np.sqrt(2 * np.abs(np.log(self.omegaLow * self.tExp))) / np.sqrt(phPar.hbar)

        return decay
