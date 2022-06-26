"""
circuit.py contains the classes for the circuit and their properties
"""

from typing import Dict, Tuple, List, Sequence, Optional, Union, Callable

import numpy as np
import qutip as qt
import scipy.special
import scipy.sparse

from numpy import ndarray
from qutip.qobj import Qobj
from scipy.linalg import sqrtm, block_diag
from scipy.special import eval_hermite

import SQcircuit.units as unt

from SQcircuit.elements import Capacitor, Inductor, Junction, Loop, Charge
from SQcircuit.texts import is_notebook, HamilTxt
from SQcircuit.noise import ENV


class Circuit:
    """
    Class that contains the circuit properties and uses the theory discussed
    in the original paper of the SQcircuit to calculate:

        * Eigenvalues and eigenvectors
        * Phase coordinate representation of eigenvectors
        * Coupling operators
        * Matrix elements
        * Decoherence rates
        * Robustness analysis

    Parameters
    ----------
        elements:
            A dictionary that contains the circuit's elements at each branch
            of the circuit.
        random:
            If `True`, each element of the circuit is a random number due to
            fabrication error. This is necessary for robustness analysis.
        flux_dist:
            Provide the method of distributing the external fluxes. If
            ``flux_dist`` is ``"all"``, SQcircuit assign the external fluxes
            based on the capacitor of each inductive element (This option is
            necessary for time-dependent external fluxes). If ``flux_dist`` is
            ``"inductor"`` SQcircuit finds the external flux distribution by
            assuming the capacitor of the inductors are much smaller than the
            junction capacitors, If ``flux_dist`` is ``"junction"`` it is the
            other way around.
    """

    def __init__(
            self,
            elements: Dict[Tuple[int, int],
                           List[Union[Capacitor, Inductor, Junction]]],
            flux_dist: str = 'junctions',
            random: bool = False
    ) -> None:

        #######################################################################
        # General circuit attributes
        #######################################################################

        self.elements = elements

        error = ("flux_dist option must be either \"junctions\", "
                 "\"inductors\", or \"all\"")
        assert flux_dist in ["junctions", "inductors", "all"], error
        self.flux_dist = flux_dist

        self.random = random

        # circuit inductive loops
        self.loops: List[Loop] = []

        # charge islands of the circuit
        self.charge_islands: Dict[int, Charge] = {}

        # number of nodes
        self.n: int = max(max(self.elements))

        # number of branches that contain JJ without parallel inductor.
        self.countJJnoInd: int = 0

        # inductor element keys: (edge, el, B_idx) B_idx point to
        # each row of B matrix (external flux distribution of that element)
        self.inductor_keys: List[tuple, Inductor, int] = []

        # junction element keys: (edge, el, B_idx, W_idx) B_idx point to
        # each row of B matrix (external flux distribution of that element)
        # and W_idx point to each row of W matrix
        self.junction_keys: List[tuple, Junction, int, int] = []

        #######################################################################
        # Transformation related attributes
        #######################################################################

        # get the capacitance matrix, sudo-inductance matrix, W matrix,
        # and B matrix (loop distribution over inductive elements)
        self.C, self.L, self.W, self.B = self._get_LCWB()

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

        # transformed sudo-inductance matrix (diagonal matrix)
        self.lTrans = np.zeros((self.n, self.n))
        # transformed capacitance matrix
        self.cTrans = np.zeros((self.n, self.n))
        # transformed inverse capacitance matrix
        self.cInvTrans = np.zeros((self.n, self.n))
        # transformed W matrix
        self.wTrans = np.zeros_like(self.W)

        # natural angular frequencies of the circuit for each mode as a numpy
        # array (zero for charge modes)
        self.omega = np.zeros(self.n)

        # transform the Hamiltonian of the circuit
        self._transform_hamil()

        #######################################################################
        # Operator and diagonalization related attributes
        #######################################################################

        # truncation numbers for each mode
        self.m = []
        # squeezed truncation numbers (eliminating the modes with truncation
        # number equals 1)
        self.ms = []

        self._memory_ops: Dict[str, Union[List[Qobj],
                                          List[List[Qobj]], dict]] = {
            "I": [],  # list of identity operators
            "Q": [],  # list of charge operators (normalized by 1/sqrt(hbar))
            "QQ": [[]],  # list of charge times charge operators
            "phi": [],  # list of flux operators (normalized by 1/sqrt(hbar))
            "N": [],  # list of number operators
            "exp": [],  # List of exponential operators
            "root_exp": [],  # List of square root of exponential operators
            "cos": {},  # List of cosine operators
            "sin": {},  # List of sine operators
            "sin_half": {},  # list of sin(phi/2)
            "ind_hamil": {},  # list of w^T*phi that appears in Hamiltonian
        }

        # LC part of the Hamiltonian
        self._LC_hamil = qt.Qobj()

        # eigenvalues of the circuit
        self._efreqs = np.array([])
        # eigenvectors of the circuit
        self._evecs = []

    def __getstate__(self):
        attrs = self.__dict__
        # type_attrs = type(self).__dict__

        # Attributes that we are avoiding to store for reducing the size of
        # the saved file( Qutip objects and Quantum operators usually).
        avoid_attrs = ["_memory_ops", "_LC_hamil"]

        self_dict = {k: attrs[k] for k in attrs if k not in avoid_attrs}

        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state

    @staticmethod
    def _independentRows(A):
        """use Gram–Schmidt to find the linear independent rows of matrix A
        """
        # normalize the row of matrix A
        A_norm = A / np.linalg.norm(A, axis=1).reshape(A.shape[0], 1)

        basis = []
        idx_list = []

        for i, a in enumerate(A_norm):
            a_prime = a - sum([np.dot(a, e) * e for e in basis])
            if (np.abs(a_prime) > 1e-7).any():
                idx_list.append(i)
                basis.append(a_prime / np.linalg.norm(a_prime))

        return idx_list, basis

    def _add_loop(self, loop: Loop) -> None:
        """
        Add loop to the circuit loops.
        """
        if loop not in self.loops:
            loop.reset()
            self.loops.append(loop)

    def _get_LCWB(self):
        """
        calculate the capacitance matrix, inductance matrix, W matrix,
        and the flux distribution over inductive elements B.
        outputs:
            -- cMat: capacitance matrix (self.n,self.n)
            -- lMat: inductance matrix (self.n,self.n)
            -- wMat:  W matrix(linear combination of the flux node operators
            in the JJ cosine (n_J,self.n)
        """

        cMat = np.zeros((self.n, self.n))
        lMat = np.zeros((self.n, self.n))
        wMat = []
        bMat = np.array([])

        # point to each row of B matrix (external flux distribution of that
        # element) or count the number of inductive elements.
        B_idx = 0

        # W_idx point to each row of W matrix for junctions or count the
        # number of edges contain JJ
        W_idx = 0

        # number of branches that contain JJ without parallel inductor.
        countJJnoInd = 0

        # K1 is a matrix that transfer node coordinates to edge phase drop
        # for inductive elements
        K1 = []
        # capacitor at each inductive elements
        cEd = []

        for edge in self.elements.keys():
            # i1 and i2 are the nodes of the edge
            i1, i2 = edge

            w = (self.n + 1) * [0]

            if i1 == 0 or i2 == 0:
                w[i1 + i2] += 1
            else:
                w[i1] += 1
                w[i2] -= 1

            # elements of the edge
            edgeElements = self.elements[edge]

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
                    # if el.loops:
                    self.inductor_keys.append((edge, el, B_idx))
                    # else:
                    #     self.inductor_keys.append((edge, el, None))
                    indList.append(el)
                    # capacitor of inductor
                    capList.append(el.cap)
                    loops = el.loops
                    for loop in loops:
                        self._add_loop(loop)
                        loop.add_index(B_idx)
                        loop.addK1(w[1:])

                    B_idx += 1
                    K1.append(w[1:])

                    if self.flux_dist == 'all':
                        cEd.append(el.cap.value())
                    elif self.flux_dist == "junctions":
                        cEd.append(Capacitor(1e20, "F").value())
                    elif self.flux_dist == "inductors":
                        cEd.append(Capacitor(1e-20, "F").value())

                elif isinstance(el, Junction):
                    # if el.loops:
                    self.junction_keys.append((edge, el, B_idx, W_idx))
                    # else:
                    #     self.junction_keys.append((edge, el, None, W_idx))
                    JJList.append(el)
                    # capacitor of JJ
                    capList.append(el.cap)
                    loops = el.loops
                    for loop in loops:
                        self._add_loop(loop)
                        loop.add_index(B_idx)
                        loop.addK1(w[1:])

                    B_idx += 1
                    K1.append(w[1:])

                    if self.flux_dist == 'all':
                        cEd.append(el.cap.value())
                    elif self.flux_dist == "junctions":
                        cEd.append(Capacitor(1e-20, "F").value())
                    elif self.flux_dist == "inductors":
                        cEd.append(Capacitor(1e20, "F").value())

            if len(indList) == 0 and len(JJList) != 0:
                countJJnoInd += 1

            # summation of the capacitor values.
            cap = sum(list(map(lambda c: c.value(self.random), capList)))

            # summation of the one over inductor values.
            x = np.sum(1 / np.array(list(map(lambda l: l.value(self.random),
                                             indList))))

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
                W_idx += 1

        wMat = np.array(wMat)

        try:
            K1 = np.array(K1)
            a = np.zeros_like(K1)
            select = np.sum(K1 != a, axis=0) != 0
            # eliminate the zero columns
            K1 = K1[:, select]
            if K1.shape[0] == K1.shape[1]:
                K1 = K1[:, 0:-1]

            X = K1.T @ np.diag(cEd)
            for loop in self.loops:
                p = np.zeros((1, B_idx))
                p[0, loop.indices] = loop.getP()
                X = np.concatenate((X, p), axis=0)

            # number of inductive loops of the circuit
            n_loops = len(self.loops)

            if n_loops != 0:
                Y = np.concatenate((np.zeros((B_idx - n_loops, n_loops)),
                                    np.eye(n_loops)), axis=0)
                bMat = np.linalg.inv(X) @ Y
                bMat = np.around(bMat, 5)

        except ValueError:

            print("The edge list does not specify a connected graph or "
                  "all inductive loops of the circuit are not specified.")

        self.countJJnoInd = countJJnoInd

        return cMat, lMat, wMat, bMat

    def _transform1(self):
        """
        First transformation of the coordinates that simultaneously diagonalizes
        the capacitance and inductance matrices.

        output:
            --  lTrans: diagonalized sudo-inductance matrix (self.n,self.n)
            --  cInvTrans: diagonalized inverse of capacitance
                matrix (self.n,self.n)
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

        cInvTrans = R1.T @ cMatInv @ R1
        lTrans = S1.T @ lMat @ S1

        lTrans[singLoc, singLoc] = 0

        return lTrans, cInvTrans, S1, R1

    def _transform2(self, omega: np.array, S1: np.array):
        """
        Second transformation of the coordinates that transforms the subspace
        of the charge operators which are defined in the charge basis in
        order to have the Bloch wave vectors in the cartesian direction.
        output:
            --  R2: Second transformation of charge operators (self.n,self.n)
            --  S2: Second transformation of flux operators (self.n,self.n)
        """

        # apply the first transformation on w and get the charge basis part
        wTrans1 = self.W @ S1
        wQ = wTrans1[:, omega == 0]

        # number of operators represented in charge bases
        nq = wQ.shape[1]

        # if we need to represent an operator in charge basis
        if nq != 0 and self.countJJnoInd != 0:

            # list of indices of w vectors that are independent
            indList = []

            X = []
            # use Gram–Schmidt to find the linear independent rows of
            # normalized wQ (wQ_norm)
            basis = []
            while len(basis) != nq:
                if len(basis) == 0:
                    indList, basis = self._independentRows(wQ)
                else:
                    # to complete the basis
                    X = list(np.random.randn(nq - len(basis), nq))
                    basisComplete = np.array(basis + X)
                    _, basis = self._independentRows(basisComplete)

            # the second S and R matrix are:
            F = np.array(list(wQ[indList, :]) + X)
            S2 = block_diag(np.eye(self.n - nq), np.linalg.inv(F))

            R2 = np.linalg.inv(S2.T)

        else:
            S2 = np.eye(self.n, self.n)
            R2 = S2

        return S2, R2

    def _transform3(self):
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
                        # check if abs(A[i,j]/s is either zero or
                        # one with 1e-11 accuracy
                        if (abs(self.wTrans[i, j] / s) >= 1e-11
                                and abs(
                                    abs(self.wTrans[i, j] / s) - 1) >= 1e-11):
                            raise ValueError("This solver cannot solve"
                                             " your circuit.")
                        if abs(self.wTrans[i, j] / s) <= 1e-11:
                            self.wTrans[i, j] = 0

                    S3[j, j] = 1 / s

                # correcting the cInvRotated values
                for i in range(self.n):
                    if i == j:
                        self.cInvTrans[i, j] = self.cInvTrans[i, j] * s ** 2
                    else:
                        self.cInvTrans[i, j] = self.cInvTrans[i, j] * s
            # for harmonic modes
            else:

                # note: alpha here is absolute value of alpha (alpha is pure
                # imaginary)

                # alpha for j-th mode
                alpha = np.abs(
                    2 * np.pi / unt.Phi0 * np.sqrt(unt.hbar / 2 * np.sqrt(
                        self.cInvTrans[j, j] / self.lTrans[
                            j, j])) * self.wTrans[:, j])

                self.wTrans[:, j][alpha < 1e-11] = 0
                if np.max(alpha) > 1e-11:
                    # find the coefficient in wTrans for j-th mode that
                    # has maximum alpha
                    s = np.abs(self.wTrans[np.argmax(alpha), j])
                    # scale that mode with s
                    self.wTrans[:, j] = self.wTrans[:, j] / s
                    S3[j, j] = 1 / s
                    for i in range(self.n):
                        if i == j:
                            self.cInvTrans[i, j] *= s ** 2
                            self.lTrans[i, j] /= s ** 2
                        else:
                            self.cInvTrans[i, j] *= s
                            self.lTrans[i, j] /= s
                else:
                    # scale the uncoupled mode
                    S = np.abs(self.S1 @ self.S2)

                    s = np.max(S[:, j])

                    S3[j, j] = 1 / s
                    for i in range(self.n):
                        if i == j:
                            self.cInvTrans[i, j] *= s ** 2
                            self.lTrans[i, j] /= s ** 2
                        else:
                            self.cInvTrans[i, j] *= s
                            self.lTrans[i, j] /= s

        R3 = np.linalg.inv(S3.T)

        return S3, R3

    def _transform_hamil(self):
        """
        transform the Hamiltonian of the circuit that can be expressed
        in charge and Fock bases
        """

        # get the first transformation:
        self.lTrans, self.cInvTrans, self.S1, self.R1 = self._transform1()
        # second transformation

        # natural frequencies of the circuit(zero for modes in charge basis)
        self.omega = np.sqrt(np.diag(self.cInvTrans) * np.diag(self.lTrans))

        # set the external charge for each charge mode.
        self.charge_islands = {i: Charge() for i in range(self.n) if
                               self.omega[i] == 0}

        # the case that circuit has no JJ
        if len(self.W) == 0:

            self.S = self.S1
            self.R = self.R1

        else:

            # get the second transformation:
            self.S2, self.R2 = self._transform2(self.omega, self.S1)

            # apply the second transformation on self.cInvTrans
            self.cInvTrans = self.R2.T @ self.cInvTrans @ self.R2

            # get the transformed W matrix
            self.wTrans = self.W @ self.S1 @ self.S2
            if self.countJJnoInd == 0:
                self.wTrans[:, self.omega == 0] = 0
            # wQ = self.wTrans[:, self.omega == 0]
            # wQ[np.abs(wQ) < 0.98] = 0
            # self.wTrans[:, self.omega == 0] = wQ

            # scaling the modes
            self.S3, self.R3 = self._transform3()

            # The final transformations are:
            self.S = self.S1 @ self.S2 @ self.S3
            self.R = self.R1 @ self.R2 @ self.R3

            # self.cTrans = np.linalg.inv(self.cInvTrans)

    def description(
            self,
            tp: Optional[str] = None,
            _test: bool = False,
    ) -> Optional[str]:
        """
        Print out Hamiltonian and a listing of the modes (whether they are
        harmonic or charge modes with the frequency for each harmonic mode),
        Hamiltonian parameters, and external flux values.

        Parameters
        ----------
            tp:
                If ``None`` prints out the output as Latex if SQcircuit is
                running in a Jupyter notebook and as text if SQcircuit is
                running in Python terminal. If ``tp`` is ``"ltx"``,
                the output is in Latex format if ``tp`` is ``"txt"`` the
                output is in text format.
            _test:
                if True, return the entire description as string
                text. (use only for testing the function)
        """
        if tp is None:
            if is_notebook():
                txt = HamilTxt('ltx')
            else:
                txt = HamilTxt('txt')
        else:
            txt = HamilTxt(tp)

        hamilTxt = txt.H()
        harDim = np.sum(self.omega != 0)
        chDim = np.sum(self.omega == 0)
        W = np.round(self.wTrans, 6)
        S = np.round(self.S, 3)
        if self.B is not None:
            B = np.round(self.B, 2)
        else:
            B = np.zeros((len(self.junction_keys) + len(self.inductor_keys), 1))
        EJLst = []
        ELLst = []

        for i in range(harDim):
            hamilTxt += txt.omega(i + 1) + txt.ad(i + 1) + \
                        txt.a(i + 1) + txt.p()

        for i in range(chDim):
            for j in range(chDim):
                if j >= i:
                    hamilTxt += txt.Ec(harDim + i + 1, harDim + j + 1) + \
                                txt.n(harDim + i + 1, harDim + j + 1) + txt.p()

        JJHamilTxt = ""
        indHamilTxt = ""

        for i, (edge, el, B_idx, W_idx) in enumerate(self.junction_keys):
            EJLst.append(el.value() / 2 / np.pi / unt.get_unit_freq())
            junTxt = txt.Ej(i + 1) + txt.cos() + "("
            # if B_idx is not None:
            junTxt += txt.linear(txt.phi, W[W_idx, :]) + \
                      txt.linear(txt.phiExt, B[B_idx, :], st=False)
            # else:
            #     junTxt += txt.linear(txt.phi, W[W_idx, :])
            JJHamilTxt += junTxt + ")" + txt.p()

        for i, (edge, el, B_idx) in enumerate(self.inductor_keys):

            # if np.sum(np.abs(B[B_idx, :])) == 0 or B_idx is None:
            if np.sum(np.abs(B[B_idx, :])) == 0:
                continue
            ELLst.append(el.energy())
            indTxt = txt.El(i + 1) + "("
            if 0 in edge:
                w = S[edge[0] + edge[1] - 1, :]
            else:
                w = S[edge[0] - 1, :] - S[edge[1] - 1, :]
            w = np.round(w[:harDim], 3)

            indTxt += txt.linear(txt.phi, w) + ")(" + \
                      txt.linear(txt.phiExt, B[B_idx, :])
            indHamilTxt += indTxt + ")" + txt.p()

        hamilTxt += indHamilTxt + JJHamilTxt

        if '+' in hamilTxt[-3:-1]:
            hamilTxt = hamilTxt[0:-2] + '\n'

        modeTxt = ''
        for i in range(harDim):
            modeTxt += txt.mode(i + 1) + txt.tab() + txt.har()

            modeTxt += txt.tab() + txt.phi(i + 1) + txt.eq() + txt.zp(i + 1) \
                       + "(" + txt.a(i + 1) + "+" + txt.ad(i + 1) + ")"

            omega = np.round(self.omega[i] / 2 / np.pi / unt.get_unit_freq(), 5)
            zp = 2 * np.pi / unt.Phi0 * np.sqrt(unt.hbar / 2 * np.sqrt(
                self.cInvTrans[i, i] / self.lTrans[i, i]))
            zpTxt = "{:.2e}".format(zp)

            modeTxt += txt.tab() + txt.omega(i + 1, False) + txt.eq() + str(
                omega) + txt.tab() + txt.zp(i + 1) + txt.eq() + zpTxt

            modeTxt += '\n'
        for i in range(chDim):
            modeTxt += txt.mode(harDim + i + 1) + txt.tab() + txt.ch()
            ng = np.round(self.charge_islands[harDim + i].value(), 3)
            modeTxt += txt.tab() + txt.ng(harDim + i + 1) + txt.eq() + str(ng)
            modeTxt += '\n'

        paramTxt = txt.param() + txt.tab()
        for i in range(chDim):
            for j in range(chDim):
                if j >= i:
                    paramTxt += txt.Ec(harDim + i + 1,
                                       harDim + j + 1) + txt.eq()

                    if i == j:
                        Ec = (2 * unt.e) ** 2 / (
                                unt.hbar * 2 * np.pi * unt.get_unit_freq()) * \
                             self.cInvTrans[
                                 harDim + i, harDim + j] / 2
                    else:
                        Ec = (2 * unt.e) ** 2 / (
                                unt.hbar * 2 * np.pi * unt.get_unit_freq()) * \
                             self.cInvTrans[
                                 harDim + i, harDim + j]

                    paramTxt += str(np.round(Ec, 3)) + txt.tab()
        for i in range(len(ELLst)):
            paramTxt += txt.El(i + 1) + txt.eq() + str(
                np.round(ELLst[i], 3)) + txt.tab()
        for i in range(len(EJLst)):
            paramTxt += txt.Ej(i + 1) + txt.eq() + str(
                np.round(EJLst[i], 3)) + txt.tab()
        paramTxt += '\n'

        loopTxt = txt.loops() + txt.tab()
        for i in range(len(self.loops)):
            phiExt = self.loops[i].value() / 2 / np.pi
            loopTxt += txt.phiExt(i + 1) + txt.tPi() + txt.eq() + str(
                phiExt) + txt.tab()

        finalTxt = hamilTxt + txt.line + modeTxt + txt.line + paramTxt + loopTxt

        txt.display(finalTxt)

        if _test:
            return finalTxt

    def loop_description(self, _test: bool = False) -> Optional[str]:
        """
        Print out the external flux distribution over inductive elements.

        Parameters
        ----------
            _test:
                if True, return the entire description as string
                text. (use only for testing the function)

        """

        # maximum length of element ID strings
        nr = max(
            [len(el.id_str) for _, el, _, _ in self.junction_keys]
            + [len(el.id_str) for _, el, _ in self.inductor_keys]
        )

        # maximum length of loop ID strings
        nh = max([len(lp.id_str) for lp in self.loops])

        # number of loops
        nl = len(self.loops)

        # space between elements in rows
        ns = 5

        loop_description_txt = ''

        header = (nr + ns + len(", b1:")) * " "
        for i in range(nl):
            lp = self.loops[i]
            header += ("{}" + (nh + 10 - len(lp.id_str)) * " ").format(
                lp.id_str)

        loop_description_txt += header + '\n'

        # add line under header
        loop_description_txt += "-" * len(header) + '\n'
        for i in range(self.B.shape[0]):

            el = None
            for _, el_ind, B_idx in self.inductor_keys:
                if i == B_idx:
                    el = el_ind
            for _, el_ind, B_idx, W_idx in self.junction_keys:
                if i == B_idx:
                    el = el_ind

            id = el.id_str
            row = id + (nr - len(id)) * " "
            bStr = f", b{i + 1}:"
            row += bStr
            row += (ns + len(", b1:") - len(bStr)) * " "
            for j in range(nl):
                b = np.round(np.abs(self.B[i, j]), 2)
                row += ("{}" + (nh + 10 - len(str(b))) * " ").format(b)
            loop_description_txt += row + '\n'

        if _test:
            return loop_description_txt
        else:
            print(loop_description_txt)

    def set_trunc_nums(self, nums: List[int]) -> None:
        """Set the truncation numbers for each mode.

        Parameters
        ----------
            nums:
                A list that contains the truncation numbers for each mode.
        """

        error1 = "The input must be be a python list"
        assert isinstance(nums, list), error1
        error2 = ("The number of modes(length of the input) must be equal to "
                  "the number of nodes")
        assert len(nums) == self.n, error2
        self.m = nums

        # squeeze the mode with truncation number equal to 1.
        self.ms = list(filter(lambda x: x != 1, self.m))

        self._build_op_memory()

        self._LC_hamil = self._get_LC_hamil()

        self._build_exp_ops()

    def set_charge_offset(self, mode: int, ng: float) -> None:
        """set the charge offset for each charge mode.

        Parameters
        ----------
            mode:
                An integer that specifies the charge mode. To see, which mode
                is a charge mode, one can use ``description()`` method.
            ng:
                The charge offset.
        """
        assert isinstance(mode, int), "Mode number should be an integer"

        error = "The specified mode is not a charge mode."
        assert mode - 1 in self.charge_islands, error
        if len(self.m) == 0:
            self.charge_islands[mode - 1].setOffset(ng)
        else:
            self.charge_islands[mode - 1].setOffset(ng)

            self._build_op_memory()

            self._LC_hamil = self._get_LC_hamil()

    def set_charge_noise(self, mode: int, A: float) -> None:
        """set the charge noise for each charge mode.

        Parameters
        ----------
            mode:
                An integer that specifies the charge mode. To see which mode
                is a charge mode, we can use ``description()`` method.
            A:
                The charge noise.
        """
        assert isinstance(mode, int), "Mode number should be an integer"

        assert mode - 1 in self.charge_islands, "The specified mode " \
                                                "is not a charge mode."

        self.charge_islands[mode - 1].setNoise(A)

    def _squeeze_op(self, op: Qobj) -> Qobj:
        """
        Return the same Quantum operator with squeezed dimensions

        Parameters
        ----------
            op:
                Any quantum operator in qutip.Qobj format
        """

        op_sq = op.copy()

        op_sq.dims = [self.ms, self.ms]

        return op_sq

    def _build_op_memory(self) -> None:
        """
        build the charge operators, number operators, and cross
        multiplication of charge operators.
        """

        charge_ops: List[Qobj] = []
        flux_ops: List[Qobj] = []
        num_ops: List[Qobj] = []
        charge_by_charge_ops: List[List[Qobj]] = []

        # list of charge operators in their own mode basis
        # (tensor product of other modes are not applied yet!)
        QList = []
        for i in range(self.n):
            if self.omega[i] == 0:
                Q0 = (2 * unt.e / np.sqrt(unt.hbar)) * \
                     (qt.charge((self.m[i] - 1) / 2)
                      - self.charge_islands[i].value())
            else:
                coef = -1j * np.sqrt(
                    0.5 * np.sqrt(self.lTrans[i, i] / self.cInvTrans[i, i]))
                Q0 = coef * (qt.destroy(self.m[i]) - qt.create(self.m[i]))
            QList.append(Q0)

        fluxList = []
        # list of flux operators in their own mode basis
        # (tensor product of other modes are not applied yet!)
        for i in range(self.n):
            if self.omega[i] == 0:
                flux0 = qt.qeye(self.m[i])
            else:
                coef = np.sqrt(0.5 * np.sqrt(self.cInvTrans[i, i] /
                                             self.lTrans[i, i]))
                flux0 = coef * (qt.destroy(self.m[i]) + qt.create(self.m[i]))
            fluxList.append(flux0)

        # list of number operators in their own mode basis
        # (tensor product of other modes are not applied yet!)
        nList = []
        for i in range(self.n):
            if self.omega[i] == 0:
                num0 = qt.charge((self.m[i] - 1) / 2)
            else:
                num0 = qt.num(self.m[i])
            nList.append(num0)

        for i in range(self.n):
            chargeRowList = []
            num = qt.Qobj()
            Q = qt.Qobj()
            flux = qt.Qobj()
            for j in range(self.n):
                # find the appropriate charge and number operator for first mode
                if j == 0 and i == 0:
                    Q2 = QList[j] * QList[j]
                    Q = QList[j]
                    num = nList[j]
                    flux = fluxList[j]

                    # Tensor product the charge with I for other modes
                    for k in range(self.n - 1):
                        Q2 = qt.tensor(Q2, qt.qeye(self.m[k + 1]))
                    chargeRowList.append(self._squeeze_op(Q2))

                elif j == 0 and i != 0:
                    I = qt.qeye(self.m[j])
                    Q = I
                    num = I
                    flux = I

                # find the rest of the modes
                elif j != 0 and j < i:
                    I = qt.qeye(self.m[j])
                    Q = qt.tensor(Q, I)
                    num = qt.tensor(num, I)
                    flux = qt.tensor(flux, I)

                elif j != 0 and j == i:
                    Q2 = qt.tensor(Q, QList[j] * QList[j])
                    Q = qt.tensor(Q, QList[j])
                    num = qt.tensor(num, nList[j])
                    flux = qt.tensor(flux, fluxList[j])

                    # Tensor product the charge with I for other modes
                    for k in range(self.n - j - 1):
                        Q2 = qt.tensor(Q2, qt.qeye(self.m[k + j + 1]))
                    chargeRowList.append(self._squeeze_op(Q2))

                elif j > i:
                    QQ = qt.tensor(Q, QList[j])

                    # Tensor product the QQ with I for other modes
                    for k in range(self.n - j - 1):
                        QQ = qt.tensor(QQ, qt.qeye(self.m[k + j + 1]))
                    chargeRowList.append(self._squeeze_op(QQ))

                    I = qt.qeye(self.m[j])
                    Q = qt.tensor(Q, I)
                    num = qt.tensor(num, I)
                    flux = qt.tensor(flux, I)

            charge_ops.append(self._squeeze_op(Q))
            charge_by_charge_ops.append(chargeRowList)
            flux_ops.append(self._squeeze_op(flux))
            num_ops.append(self._squeeze_op(num))

        self._memory_ops["Q"] = charge_ops
        self._memory_ops["QQ"] = charge_by_charge_ops
        self._memory_ops["phi"] = flux_ops
        self._memory_ops["N"] = num_ops

    def _get_LC_hamil(self) -> Qobj:
        """
        get the LC part of the Hamiltonian
        outputs:
            -- HLC: LC part of the Hamiltonian (qutip Object)
        """

        LC_hamil = qt.Qobj()

        for i in range(self.n):
            # we write j in this form because of "_memory_ops["QQ"]" shape
            for j in range(self.n - i):
                if j == 0:
                    if self.omega[i] == 0:
                        LC_hamil += (0.5 * self.cInvTrans[i, i]
                                     * self._memory_ops["QQ"][i][j])
                    else:
                        LC_hamil += self.omega[i] * self._memory_ops["N"][i]

                elif j > 0:
                    if self.cInvTrans[i, i + j] != 0:
                        LC_hamil += (self.cInvTrans[i, i + j]
                                     * self._memory_ops["QQ"][i][j])

        return LC_hamil

    @staticmethod
    def _d_op(N: int) -> Qobj:
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
        d = qt.Qobj(d)
        d = d.dag()

        return d

    def _build_exp_ops(self) -> None:
        """
        Each cosine potential of the Josephson Junction can be written as
        summation of two exponential terms,cos(x)=(exp(ix)+exp(-ix))/2. This
        function returns the quantum operators for only one exponential term.
        """

        exp_ops = []
        root_exp_ops = []

        # number of Josephson Junctions
        nJ = self.wTrans.shape[0]

        H = 0
        # for calculating sin(phi/2) operator for quasi-particle
        # loss decay rate
        H2 = 0

        for i in range(nJ):

            # tensor multiplication of displacement operator for JJ Hamiltonian
            for j in range(self.n):
                if j == 0 and self.omega[j] == 0:
                    if self.wTrans[i, j] == 0:
                        I = qt.qeye(self.m[j])
                        H = I
                        H2 = I
                    elif self.wTrans[i, j] > 0:
                        d = self._d_op(self.m[j])
                        I = qt.qeye(self.m[j])
                        H = d
                        # not correct just to avoid error:
                        H2 = I
                    else:
                        d = self._d_op(self.m[j])
                        I = qt.qeye(self.m[j])
                        H = d.dag()
                        # not correct just to avoid error:
                        H2 = I

                elif j == 0 and self.omega[j] != 0:
                    alpha = 2 * np.pi / unt.Phi0 * 1j * np.sqrt(
                        unt.hbar / 2 * np.sqrt(
                            self.cInvTrans[j, j] / self.lTrans[j, j])) * \
                            self.wTrans[i, j]
                    H = qt.displace(self.m[j], alpha)
                    H2 = qt.displace(self.m[j], alpha / 2)

                if j != 0 and self.omega[j] == 0:
                    if self.wTrans[i, j] == 0:
                        I = qt.qeye(self.m[j])
                        H = qt.tensor(H, I)
                        H2 = qt.tensor(H2, I)
                    elif self.wTrans[i, j] > 0:
                        I = qt.qeye(self.m[j])
                        d = self._d_op(self.m[j])
                        H = qt.tensor(H, d)
                        H2 = qt.tensor(H2, I)
                    else:
                        I = qt.qeye(self.m[j])
                        d = self._d_op(self.m[j])
                        H = qt.tensor(H, d.dag())
                        H2 = qt.tensor(H2, I)

                elif j != 0 and self.omega[j] != 0:
                    alpha = 2 * np.pi / unt.Phi0 * 1j * np.sqrt(
                        unt.hbar / 2 * np.sqrt(
                            self.cInvTrans[j, j] / self.lTrans[j, j])) * \
                            self.wTrans[i, j]
                    H = qt.tensor(H, qt.displace(self.m[j], alpha))
                    H2 = qt.tensor(H2, qt.displace(self.m[j], alpha / 2))

            exp_ops.append(self._squeeze_op(H))
            root_exp_ops.append(self._squeeze_op(H2))

        self._memory_ops["exp"] = exp_ops
        self._memory_ops["root_exp"] = root_exp_ops

    def _get_external_flux_at_element(self, B_idx: int) -> float:
        """
        Return the external flux at an inductive element.

        Parameters
        ----------
            B_idx:
                An integer point to each row of B matrix (external flux
                distribution of that element)
        """
        phi_ext = 0.0
        for i, loop in enumerate(self.loops):
            phi_ext += loop.value(self.random) * self.B[B_idx, i]

        return phi_ext

    def _get_inductive_hamil(self) -> Qobj:

        H = qt.Qobj()

        for edge, el, B_idx in self.inductor_keys:
            # phi = 0
            # if B_idx is not None:
            phi = self._get_external_flux_at_element(B_idx)

            # summation of the 1 over inductor values.
            x = 1 / el.value(self.random)
            O = self.coupling_op("inductive", edge)
            H += x * phi * (unt.Phi0 / 2 / np.pi) * O / np.sqrt(unt.hbar)

            # save the operators for loss calculation
            self._memory_ops["ind_hamil"][(el, B_idx)] = O

        for _, el, B_idx, W_idx in self.junction_keys:
            # phi = 0
            # if B_idx is not None:
            phi = self._get_external_flux_at_element(B_idx)

            EJ = el.value(self.random)

            exp = np.exp(1j * phi) * self._memory_ops["exp"][W_idx]
            root_exp = np.exp(1j * phi / 2) * self._memory_ops["root_exp"][
                W_idx]

            cos = (exp + exp.dag()) / 2
            sin = (exp - exp.dag()) / 2j
            sin_half = (root_exp - root_exp.dag()) / 2j

            self._memory_ops["cos"][el, B_idx] = self._squeeze_op(cos)
            self._memory_ops["sin"][el, B_idx] = self._squeeze_op(sin)
            self._memory_ops["sin_half"][el, B_idx] = self._squeeze_op(sin_half)

            H += -EJ * cos

        return H

    def diag(self, n_eig: int) -> Tuple[ndarray, List[Qobj]]:
        """
        Diagonalize the Hamiltonian of the circuit and return the
        eigenfrequencies and eigenvectors of the circuit up to specified
        number of eigenvalues.

        Parameters
        ----------
            n_eig:
                Number of eigenvalues to output. The lower ``n_eig``, the
                faster ``SQcircuit`` finds the eigenvalues.
        Returns
        ----------
            efreq:
                ndarray of eigenfrequencies in frequency unit of SQcircuit (
                gigahertz by default)
            evecs:
                List of eigenvectors in qutip.Qobj format.
        """
        error1 = "Please specify the truncation number for each mode."
        assert len(self.m) != 0, error1
        error2 = "n_eig (number of eigenvalues) should be an integer."
        assert isinstance(n_eig, int), error2

        H = self.hamiltonian()

        # get the data out of qutip variable and use sparse scipy eigen
        # solver which is faster.
        efreqs, evecs = scipy.sparse.linalg.eigs(H.data, n_eig, which='SR')
        # the output of eigen solver is not sorted
        efreqs_sorted = np.sort(efreqs.real)

        sort_arg = np.argsort(efreqs)
        if isinstance(sort_arg, int):
            sort_arg = [sort_arg]

        evecs_sorted = [
            qt.Qobj(evecs[:, ind], dims=[self.ms, len(self.ms) * [1]])
            for ind in sort_arg
        ]

        # store the eigenvalues and eigenvectors of the circuit Hamiltonian
        self._efreqs = efreqs_sorted
        self._evecs = evecs_sorted

        return efreqs_sorted / (2*np.pi*unt.get_unit_freq()), evecs_sorted

    ###########################################################################
    # Methods that calculate circuit properties
    ###########################################################################

    def coord_transform(self, var_type: str) -> ndarray:
        """
        Return the transformation of the coordinates as ndarray for each type
        of variables, either charge or flux.

        Parameters
        ----------
            var_type:
                The type of the variables that can be either ``"charge"`` or
                ``"flux"``.
        """
        if var_type == "charge" or var_type == "Charge":
            return np.linalg.inv(self.R)
        elif var_type == "flux" or var_type == "Flux":
            return np.linalg.inv(self.S)
        else:
            raise ValueError("The input must be either \"charge\" or \"flux\".")

    def hamiltonian(self) -> Qobj:
        """
        Returns the transformed hamiltonian of the circuit as
        qutip.Qobj format.
        """
        error = "Please specify the truncation number for each mode."
        assert len(self.m) != 0, error

        Hind = self._get_inductive_hamil()

        H = Hind + self._LC_hamil

        return H

    def _tensor_to_modes(self, tensorIndex: int) -> List[int]:
        """
        decomposes the tensor product space index to each mode indices. For
        example index 5 of the tensor product space can be decomposed to [1,
        0,1] modes if the truncation number for each mode is 2.
        inputs:
            -- tensorIndex: Index of tensor product space
        outputs:
            -- indList: a list of mode indices (self.n)
        """

        # i-th mP element is the multiplication of the self.m elements until
        # its i-th element
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

    def eig_phase_coord(self, k: int, grid: Sequence[ndarray]) -> ndarray:
        """
        Return the phase coordinate representations of the eigenvectors as
        ndarray.

        Parameters
        ----------
            k:
                The eigenvector index. For example, we set it to 0 for the
                ground state and 1 for the first excited state.
            grid:
                A list that contains the range of values of phase φ for which
                we want to evaluate the wavefunction.
        """

        assert isinstance(k, int), ("The k (index of eigenstate) should be "
                                    "an integer.")

        phi_list = [*np.meshgrid(*grid, indexing='ij')]

        # The total dimension of the circuit Hilbert Space
        netDimension = np.prod(self.m)

        state = 0

        for i in range(netDimension):

            # decomposes the tensor product space index (i) to each mode
            # indices as a list
            indList = self._tensor_to_modes(i)

            term = self._evecs[k][i][0, 0]

            for mode in range(self.n):

                # mode number related to that node
                n = indList[mode]

                # For charge basis
                if self.omega[mode] == 0:
                    term *= 1 / np.sqrt(2 * np.pi) * np.exp(
                        1j * phi_list[mode] * n)
                # For harmonic basis
                else:
                    x0 = np.sqrt(unt.hbar * np.sqrt(
                        self.cInvTrans[mode, mode] / self.lTrans[mode, mode]))

                    coef = 1 / np.sqrt(np.sqrt(np.pi) * (2 ** n) *
                                       scipy.special.factorial(
                                           n) * x0 / unt.Phi0)

                    term *= coef * np.exp(
                        -(phi_list[mode]*unt.Phi0/x0)**2/2) * \
                            eval_hermite(n, phi_list[mode] * unt.Phi0 / x0)

            state += term

        state = np.squeeze(state)

        # transposing the first two modes
        if len(state.shape) > 1:
            indModes = list(range(len(state.shape)))
            indModes[0] = 1
            indModes[1] = 0
            state = state.transpose(*indModes)

        return state

    def coupling_op(
            self,
            ctype: str,
            nodes: Tuple[int, int]
    ) -> Qobj:
        """
        Return the capacitive or inductive coupling operator related to the
        specified nodes. The output has the `qutip.Qobj` format.

        Parameters
        ----------
            ctype:
                Coupling type which is either ``"capacitive"`` or
                ``"inductive"``.
            nodes:
                A tuple of circuit nodes to which we want to couple.
        """
        error1 = ("The coupling type must be either \"capacitive\" or "
                  "\"inductive\"")
        assert ctype in ["capacitive", "inductive"], error1
        error2 = "Nodes must be a tuple of int"
        assert isinstance(nodes, tuple) or isinstance(nodes, list), error2

        op = qt.Qobj()

        node1 = nodes[0]
        node2 = nodes[1]

        # for the case that we have ground in the edge
        if 0 in nodes:
            node = node1 + node2
            if ctype == "capacitive":
                # K = np.linalg.inv(self.getMatC()) @ self.R
                K = np.linalg.inv(self.C) @ self.R
                for i in range(self.n):
                    op += K[node - 1, i] * self._memory_ops["Q"][i]
            if ctype == "inductive":
                K = self.S
                for i in range(self.n):
                    op += K[node - 1, i] * self._memory_ops["phi"][i]

        else:
            if ctype == "capacitive":
                # K = np.linalg.inv(self.getMatC()) @ self.R
                K = np.linalg.inv(self.C) @ self.R
                for i in range(self.n):
                    op += (K[node2 - 1, i] - K[node1 - 1, i]) * \
                          self._memory_ops["Q"][i]
            if ctype == "inductive":
                K = self.S
                for i in range(self.n):
                    op += ((K[node1 - 1, i] - K[node2 - 1, i])
                           * self._memory_ops["phi"][i])

        return self._squeeze_op(op)

    def matrix_elements(
            self,
            ctype: str,
            nodes: Tuple[int, int],
            states: Tuple[int, int],
    ) -> float:
        """
        Return the matrix element of two eigenstates for either capacitive
        or inductive coupling.

        Parameters
        ----------
            ctype:
                Coupling type which is either ``"capacitive"`` or
                ``"inductive"``.
            nodes:
                A tuple of circuit nodes to which we want to couple.
            states:
                A tuple of indices of eigenstates for which we want to
                calculate the matrix element.
        """

        state1 = self._evecs[states[0]]
        state2 = self._evecs[states[1]]

        # get the coupling operator
        op = self.coupling_op(ctype, nodes)

        return (state1.dag() * op * state2).data[0, 0]

    @staticmethod
    def _dephasing(A: float, partial_omega: float) -> float:
        """
        calculate dephasing rate.

        Parameters
        ----------
            A:
                Noise Amplitude
            partial_omega:
                The derivatives of angular frequency with respect to the
                noisy parameter
        """

        return (np.abs(partial_omega * A)
                * np.sqrt(2 * np.abs(np.log(ENV["omega_low"] * ENV["t_exp"]))))

    def dec_rate(
            self,
            dec_type: str,
            states: Tuple[int, int],
            total: bool = True
    ) -> float:
        """ Return the decoherence rate in [1/s] between each two eigenstates
        for different types of depolarization and dephasing.

        Parameters
        ----------
            dec_type:
                decoherence type that can be: ``"capacitive"`` for capacitive
                loss; ``"inductive"`` for inductive loss; `"quasiparticle"` for
                quasiparticle loss; ``"charge"`` for charge noise, ``"flux"``
                for flux noise; and ``"cc"`` for critical current noise.
            states:
                A tuple of eigenstate indices, for which we want to
                calculate the decoherence rate. For example, for ``states=(0,
                1)``, we calculate the decoherence rate between the ground
                state and the first excited state.
            total:
                if False return a decoherence rate associated with a
                transition from state m to state n for ``states=(m, n)``. if
                True return a decoherence rate associated with both m to n
                and n to m transitions.

        """

        omega1 = self._efreqs[states[0]]
        omega2 = self._efreqs[states[1]]

        state1 = self._evecs[states[0]]
        state2 = self._evecs[states[1]]

        omega = np.abs(omega2 - omega1)

        decay = 0

        # prevent the exponential overflow(exp(709) is the biggest number
        # that numpy can calculate
        if unt.hbar * omega / (unt.k_B * ENV["T"]) > 709:
            down = 2
            up = 0
        else:
            alpha = unt.hbar * omega / (unt.k_B * ENV["T"])
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

        if dec_type == "capacitive":
            for edge in self.elements.keys():
                for el in self.elements[edge]:
                    if isinstance(el, Capacitor):
                        cap = el
                    else:
                        cap = el.cap
                    if cap.Q:
                        decay += tempS * cap.value() / cap.Q(omega) * np.abs(
                            self.matrix_elements(
                                "capacitive", edge, states)) ** 2

        if dec_type == "inductive":
            for el, _ in self._memory_ops["ind_hamil"]:
                op = self._memory_ops["ind_hamil"][(el, _)]
                x = 1 / el.value()
                if el.Q:
                    decay += tempS / el.Q(omega, ENV["T"]) * x * np.abs(
                        (state1.dag() * op * state2).data[0, 0]) ** 2

        if dec_type == "quasiparticle":
            for el, _ in self._memory_ops['sin_half']:
                op = self._memory_ops['sin_half'][(el, _)]
                decay += tempS * el.Y(omega, ENV["T"]) * omega * el.value() \
                         * unt.hbar * np.abs(
                    (state1.dag() * op * state2).data[0, 0]) ** 2

        elif dec_type == "charge":
            # first derivative of the Hamiltonian with respect to charge noise
            op = qt.Qobj()
            for i in range(self.n):
                if self.omega[i] == 0:
                    for j in range(self.n):
                        op += (self.cInvTrans[i, j] * self._memory_ops["Q"][j]
                               / np.sqrt(unt.hbar))
                    partial_omega = np.abs((state2.dag()*op*state2 -
                                            state1.dag()*op*state1).data[0, 0])
                    A = (self.charge_islands[i].A * 2 * unt.e)
                    decay += self._dephasing(A, partial_omega)

        elif dec_type == "cc":
            for el, _ in self._memory_ops['cos']:
                op = el.value(self.random) * self._memory_ops['cos'][(el, _)]
                partial_omega = np.abs((state2.dag()*op*state2
                                        - state1.dag()*op*state1).data[0, 0])
                A = el.A
                decay += self._dephasing(A, partial_omega)

        elif dec_type == "flux":
            for loop in self.loops:
                partial_omega = self._get_partial_omega(loop, states=states)
                A = loop.A
                decay += self._dephasing(A, partial_omega)

        return decay

    def _get_partial_H(
            self,
            el: Union[Capacitor, Inductor, Junction, Loop]
    ) -> Qobj:
        """
        return the gradient of the Hamiltonian with respect to elements or
        loop as ``qutip.Qobj`` format.

        Parameters
        ----------
            el:
                element of a circuit that can be either ``Capacitor``,
                ``Inductor``, ``Junction``, or ``Loop``.
        """

        partial_H = qt.Qobj()

        if isinstance(el, Loop):

            loop_idx = self.loops.index(el)
            # note that this is not b_i
            # k = self.B[:, idx]

            for edge, el_ind, B_idx in self.inductor_keys:
                partial_H += (self.B[B_idx, loop_idx]
                              * self._memory_ops["ind_hamil"][(el_ind, B_idx)]
                              / el_ind.value() * unt.Phi0 / np.sqrt(unt.hbar)
                              / 2 / np.pi)

            for edge, el_ind, B_idx, W_idx in self.junction_keys:
                partial_H += (self.B[B_idx, loop_idx] * el_ind.value()
                              * self._memory_ops['sin'][(el_ind, B_idx)])

        return partial_H

    def _get_partial_omega(
            self,
            el: Union[Capacitor, Inductor, Junction, Loop],
            states: Tuple[int, int]
    ) -> float:
        """
        return the gradient of the eigen angular frequency with respect to
        elements or loop as ``qutip.Qobj`` format. Note that if
        ``states=(m, n)``, it returns ``partial_omega_m - partial_omega_n``.

        Parameters
        ----------
            el:
                element of a circuit that can be either ``Capacitor``,
                ``Inductor``, ``Junction``, or ``Loop``.
            states:
                A tuple of eigenstate indices, for which we want to
                calculate the decoherence rate. For example, for ``states=(0,
                1)``, we calculate the decoherence rate between the ground
                state and the first excited state.

        """
        state_m = self._evecs[states[0]]
        state_n = self._evecs[states[1]]

        partial_H = self._get_partial_H(el)

        partial_omega_m = state_m.dag() * (partial_H * state_m)
        partial_omega_n = state_n.dag() * (partial_H * state_n)

        return (partial_omega_m - partial_omega_n).data[0, 0].real
