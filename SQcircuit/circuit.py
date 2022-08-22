"""circuit.py contains the classes for the circuit and their properties
"""

from typing import Dict, Tuple, List, Sequence, Optional, Union

import numpy as np
import qutip as qt
import scipy.special
import scipy.sparse

from numpy import ndarray
from qutip.qobj import Qobj
from scipy.linalg import sqrtm, block_diag
from scipy.special import eval_hermite

import SQcircuit.units as unt

from SQcircuit.elements import (Capacitor, Inductor, Junction, Loop, Charge,
                                VerySmallCap, VeryLargeCap)
from SQcircuit.texts import is_notebook, HamilTxt
from SQcircuit.noise import ENV
from SQcircuit.settings import ACC


class Circuit:
    """Class that contains the circuit properties and uses the theory discussed
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

        # # charge islands of the circuit
        # self.charge_islands: Dict[int, Charge] = {}

        # number of nodes without ground
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
        (self.C, self.L, self.W, self.B,
         self.partial_C, self.partial_L) = self._get_LCWB()

        # initialize the transformation matrix for charge and flux operators.
        self.R, self.S = np.eye(self.n), np.eye(self.n)

        # initialize transformed susceptance, inverse capacitance, and W matrix.
        self.cInvTrans, self.lTrans, self.wTrans = (np.linalg.inv(self.C),
                                                    self.L.copy(),
                                                    self.W.copy())

        # natural angular frequencies of the circuit for each mode as a numpy
        # array (zero for charge modes)
        self.omega = np.zeros(self.n)

        # transform the Hamiltonian of the circuit
        self._transform_hamil()

        # charge islands of the circuit
        self.charge_islands = {i: Charge() for i in range(self.n) if
                               self._is_charge_mode(i)}

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

    @property
    def efreqs(self):

        assert len(self._efreqs) != 0, "Please diagonalize the circuit first."

        return self._efreqs / (2*np.pi*unt.get_unit_freq())

    def _add_loop(self, loop: Loop) -> None:
        """Add loop to the circuit loops.
        """
        if loop not in self.loops:
            loop.reset()
            self.loops.append(loop)

    def _get_w_at_edge(self, edge: Tuple[int, int]) -> list:
        """Get the w_k vector as list at the edge.

        Parameters
        ----------
            edge:
                Tuple of int which specifies an edge.
        """

        # i1 and i2 are the nodes of the edge
        i1, i2 = edge

        w = (self.n + 1) * [0]

        if i1 == 0 or i2 == 0:
            w[i1 + i2] += 1
        else:
            w[i1] += 1
            w[i2] -= 1

        return w[1:]

    def _edge_matrix_rep(self, edge: Tuple[int, int]) -> ndarray:

        """Special form of matrix representation for an edge of a graph.
        This helps to construct the capacitance and susceptance matrices.

        Parameters
        ----------
            edge:
                Tuple of int which specifies an edge.
        """

        A = np.zeros((self.n, self.n))

        if 0 in edge:
            i = edge[0] + edge[1] - 1
            A[i, i] = 1
        else:
            i = edge[0] - 1
            j = edge[1] - 1

            A[i, i] = 1
            A[j, j] = 1
            A[i, j] = -1
            A[j, i] = -1

        return A

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

        partial_cMats: Dict[Tuple[Capacitor, ndarray]] = {}

        partial_lMats: Dict[Tuple[Inductor, ndarray]] = {}

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

            # w vector at the edge
            edge_w = self._get_w_at_edge(edge)

            # matrix representation of an edge
            edge_mat = self._edge_matrix_rep(edge)

            # list of capacitors of the edge.
            edge_caps = []
            # list of inductors of the edge
            edge_inds = []
            # list of Josephson Junction of the edge.
            edge_JJs = []

            for el in self.elements[edge]:

                if isinstance(el, Capacitor):
                    edge_caps.append(el)

                    if el in partial_cMats:
                        partial_cMats[el] += edge_mat
                    else:
                        partial_cMats[el] = edge_mat

                elif isinstance(el, Inductor):
                    # if el.loops:
                    self.inductor_keys.append((edge, el, B_idx))
                    # else:
                    #     self.inductor_keys.append((edge, el, None))
                    edge_inds.append(el)
                    # capacitor of inductor
                    edge_caps.append(el.cap)

                    for loop in el.loops:
                        self._add_loop(loop)
                        loop.add_index(B_idx)
                        loop.addK1(edge_w)

                    B_idx += 1
                    K1.append(edge_w)

                    if self.flux_dist == 'all':
                        cEd.append(el.cap.value())
                    elif self.flux_dist == "junctions":
                        cEd.append(VeryLargeCap().value())
                    elif self.flux_dist == "inductors":
                        cEd.append(VerySmallCap().value())

                    if el in partial_lMats:
                        partial_lMats[el] += edge_mat / el.value()**2
                    else:
                        partial_lMats[el] = edge_mat / el.value()**2

                elif isinstance(el, Junction):
                    # if el.loops:
                    self.junction_keys.append((edge, el, B_idx, W_idx))
                    # else:
                    #     self.junction_keys.append((edge, el, None, W_idx))
                    edge_JJs.append(el)
                    # capacitor of JJ
                    edge_caps.append(el.cap)

                    for loop in el.loops:
                        self._add_loop(loop)
                        loop.add_index(B_idx)
                        loop.addK1(edge_w)

                    B_idx += 1
                    K1.append(edge_w)

                    if self.flux_dist == 'all':
                        cEd.append(el.cap.value())
                    elif self.flux_dist == "junctions":
                        cEd.append(VerySmallCap().value())
                    elif self.flux_dist == "inductors":
                        cEd.append(VeryLargeCap().value())

            if len(edge_inds) == 0 and len(edge_JJs) != 0:
                countJJnoInd += 1

            # summation of the capacitor values.
            cap = sum(list(map(lambda c: c.value(self.random), edge_caps)))

            # summation of the one over inductor values.
            x = np.sum(1 / np.array(list(map(lambda l: l.value(self.random),
                                             edge_inds))))

            cMat += cap * edge_mat

            lMat += x * edge_mat

            if len(edge_JJs) != 0:
                wMat.append(edge_w)
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

        return cMat, lMat, wMat, bMat, partial_cMats, partial_lMats

    def _is_charge_mode(self, i: int) -> bool:
        """Check if the mode is a charge mode.

        Parameters
        ----------
            i:
                index of the mode. (starts from zero for the first mode)
        """

        return self.omega[i] == 0

    def _apply_transformation(self, S: ndarray, R: ndarray) -> None:
        """Apply S and R transformation on transformed C, L, and W matrix.

        Parameters
        ----------
            S:
                Transformation matrices related to flux operators.
            R:
                Transformation matrices related to charge operators.
        """

        self.cInvTrans = R.T @ self.cInvTrans @ R
        self.lTrans = S.T @ self.lTrans @ S

        if len(self.W) != 0:
            self.wTrans = self.wTrans @ S

        self.S = self.S @ S
        self.R = self.R @ R

    def _get_and_apply_transformation_1(self) -> Tuple[ndarray, ndarray]:
        """Get and apply Second transformation of the coordinates that
        simultaneously diagonalizes the capacitance and susceptance matrices.
        """

        cMatRoot = sqrtm(self.C)
        cMatRootInv = np.linalg.inv(cMatRoot)
        lMatRoot = sqrtm(self.L)

        V, D, U = np.linalg.svd(lMatRoot @ cMatRootInv)

        # the case that there is not any inductor in the circuit
        if np.max(D) == 0:
            D = np.diag(np.eye(self.n))
            singLoc = list(range(0, self.n))
        else:
            # find the number of singularity in the circuit
            lEig, _ = np.linalg.eig(self.L)
            numSing = len(lEig[lEig / np.max(lEig) < ACC["sing_mode_detect"]])
            singLoc = list(range(self.n - numSing, self.n))
            D[singLoc] = np.max(D)

        # build S1 and R1 matrix
        S1 = cMatRootInv @ U.T @ np.diag(np.sqrt(D))
        R1 = np.linalg.inv(S1).T

        self._apply_transformation(S1, R1)

        self.lTrans[singLoc, singLoc] = 0

        return S1, R1

    @staticmethod
    def _independentRows(A: ndarray) -> Tuple[List[int], List[ndarray]]:
        """Use Gram–Schmidt to find the linear independent rows of matrix A
        and return the list of row indices of A and list of the rows.

        Parameters
        ----------
            A:
                ``Numpy.ndarray`` matrix that we try to find its independent
                rows.
        """

        # normalize the row of matrix A
        A_norm = A / np.linalg.norm(A, axis=1).reshape(A.shape[0], 1)

        basis = []
        idx_list = []

        for i, a in enumerate(A_norm):
            a_prime = a - sum([np.dot(a, e) * e for e in basis])
            if (np.abs(a_prime) > ACC["Gram–Schmidt"]).any():
                idx_list.append(i)
                basis.append(a_prime / np.linalg.norm(a_prime))

        return idx_list, basis

    def _round_to_zero_one(self, W: ndarray) -> ndarray:
        """Round the charge mode elements of W or transformed W matrix that
        are close to 0, -1, and 1 to the exact value of 0, -1, and 1
        respectively.

        Parameters
        ----------
            W:
                ``Numpy.ndarray`` that can be either W or transformed W matrix.
        """

        rounded_W = W.copy()

        if self.countJJnoInd == 0:
            rounded_W[:, self.omega == 0] = 0

        charge_only_W = rounded_W[:, self.omega == 0]

        charge_only_W[np.abs(charge_only_W) < ACC["Gram–Schmidt"]] = 0
        charge_only_W[np.abs(charge_only_W - 1) < ACC["Gram–Schmidt"]] = 1
        charge_only_W[np.abs(charge_only_W + 1) < ACC["Gram–Schmidt"]] = -1

        rounded_W[:, self.omega == 0] = charge_only_W

        return rounded_W

    def _is_JJ_in_circuit(self) -> bool:
        """Check if there is any Josephson junction in the circuit."""

        return len(self.W) != 0

    def _get_and_apply_transformation_2(self) -> Tuple[ndarray, ndarray]:
        """ Get and apply Second transformation of the coordinates that
        transforms the subspace of the charge operators in order to have the
        reciprocal primitive vectors in Cartesian direction.
        """

        if len(self.W) != 0:
            # get the charge basis part of the wTrans matrix
            wQ = self.wTrans[:, self.omega == 0].copy()
            # number of operators represented in charge bases
            nq = wQ.shape[1]
        else:
            nq = 0
            wQ = np.array([])

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

        if self._is_Gram_Schmidt_successful(S2):

            self._apply_transformation(S2, R2)

            self.wTrans = self._round_to_zero_one(self.wTrans)

            return S2, R2

        else:
            print("Gram_Schmidt process failed. Retrying...")

            return self._get_and_apply_transformation_2()

    def _is_Gram_Schmidt_successful(self, S) -> bool:
        """Check if the Gram_Schmidt process has the sufficient accuracy.

        Parameters
        ----------
            S:
                Transformation matrices related to flux operators.
        """

        is_successful = True

        # absolute value of the current wTrans
        cur_wTrans = self.wTrans @ S

        cur_wTrans = self._round_to_zero_one(cur_wTrans)

        for j in range(self.n):
            if self._is_charge_mode(j):
                for abs_w in np.abs(cur_wTrans[:, j]):
                    if abs_w != 0 and abs_w != 1:
                        is_successful = False

        return is_successful

    def _get_and_apply_transformation_3(self) -> Tuple[ndarray, ndarray]:
        """ Get and apply Third transformation of the coordinates that scales
        the modes.
        """

        S3 = np.eye(self.n)

        for j in range(self.n):

            if self._is_charge_mode(j):
                # already scaled by second transformation
                continue

            # for harmonic modes
            elif self._is_JJ_in_circuit():

                # note: alpha here is absolute value of alpha (alpha is pure
                # imaginary)
                # get alpha for j-th mode
                jth_alphas = np.abs(self.alpha(range(self.wTrans.shape[0]), j))
                self.wTrans[:, j][jth_alphas < ACC["har_mode_elim"]] = 0

                if np.max(jth_alphas) > ACC["har_mode_elim"]:
                    # find the coefficient in wTrans for j-th mode that
                    # has maximum alpha
                    s = np.abs(self.wTrans[np.argmax(jth_alphas), j])
                    S3[j, j] = 1 / s
                else:
                    # scale the uncoupled mode
                    s = np.max(np.abs(self.S[:, j]))
                    S3[j, j] = 1 / s

            else:
                # scale the uncoupled mode
                s = np.max(np.abs(self.S[:, j]))
                S3[j, j] = 1 / s

        R3 = np.linalg.inv(S3.T)

        self._apply_transformation(S3, R3)

        return S3, R3

    def _transform_hamil(self):
        """transform the Hamiltonian of the circuit that can be expressed
        in charge and Fock bases
        """

        # get the first transformation
        self.S1, self.R1 = self._get_and_apply_transformation_1()

        # natural frequencies of the circuit(zero for modes in charge basis)
        self.omega = np.sqrt(np.diag(self.cInvTrans) * np.diag(self.lTrans))

        if self._is_JJ_in_circuit():
            # get the second transformation
            self.S2, self.R2 = self._get_and_apply_transformation_2()

        # scaling the modes by third transformation
        self.S3, self.R3 = self._get_and_apply_transformation_3()

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

        # If circuit has any loop:
        if self.loops:
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
                Harmonic modes with truncation number N are 0, 1 , ...,
                (N-1), and charge modes with truncation number N are -(N-1),
                ..., 0, ..., (N-1).
        """

        error1 = "The input must be be a python list"
        assert isinstance(nums, list), error1
        error2 = ("The number of modes (length of the input) must be equal to "
                  "the number of nodes")
        assert len(nums) == self.n, error2

        self.m = self.n*[1]

        for i in range(self.n):
            # for charge modes:
            if self._is_charge_mode(i):
                self.m[i] = 2 * nums[i] - 1
            # for harmonic modes
            else:
                self.m[i] = nums[i]

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
        Return the same Quantum operator with squeezed dimensions.

        Parameters
        ----------
            op:
                Any quantum operator in qutip.Qobj format
        """

        op_sq = op.copy()

        op_sq.dims = [self.ms, self.ms]

        return op_sq

    def _charge_op_isolated(self, i: int) -> Qobj:
        """Return charge operator for each isolated mode normalized by
        square root of hbar. By isolated, we mean that the operator is not in
        the general tensor product states of the overall system.

        Parameters
        ----------
            i:
                Index of the mode. (starts from zero for the first mode)
        """

        if self._is_charge_mode(i):
            ng = self.charge_islands[i].value()
            op = (2*unt.e/np.sqrt(unt.hbar)) * (qt.charge((self.m[i]-1)/2)-ng)

        else:
            Z = np.sqrt(self.cInvTrans[i, i] / self.lTrans[i, i])
            Q_zp = -1j * np.sqrt(0.5/Z)
            op = Q_zp * (qt.destroy(self.m[i]) - qt.create(self.m[i]))

        return op

    def _flux_op_isolated(self, i: int) -> Qobj:
        """Return flux operator for each isolated mode normalized by
        square root of hbar. By isolated, we mean that the operator is not in
        the general tensor product states of the overall system.

        Parameters
        ----------
            i:
                Index of the mode. (starts from zero for the first mode)
        """

        if self._is_charge_mode(i):
            op = qt.qeye(self.m[i])

        else:
            Z = np.sqrt(self.cInvTrans[i, i] / self.lTrans[i, i])
            op = np.sqrt(0.5*Z) * (qt.destroy(self.m[i])+qt.create(self.m[i]))

        return op

    def _num_op_isolated(self, i: int) -> Qobj:
        """Return number operator for each isolated mode. By isolated,
        we mean that the operator is not in the general tensor product states
        of the overall system.

        Parameters
        ----------
            i:
                Index of the mode. (starts from zero for the first mode)
        """

        if self._is_charge_mode(i):
            op = qt.charge((self.m[i] - 1) / 2)

        else:
            op = qt.num(self.m[i])

        return op

    def _d_op_isolated(self, i: int, w: float) -> Qobj:
        """Return charge displacement operator for each isolated mode. By
        isolated, we mean that the operator is not in the general tensor
        product states of the overall system.

        Parameters
        ----------
            i:
                Index of the mode. (starts from zero for the first mode)
            w:
                Represent the power of the displacement operator, d^w. Right
                now w should be only 0, 1, and -1.
        """

        if w == 0:
            return qt.qeye(self.m[i])

        d = np.zeros((self.m[i], self.m[i]))

        for k in range(self.m[i]):
            for j in range(self.m[i]):
                if j - 1 == k:
                    d[k, j] = 1
        d = qt.Qobj(d)

        if w < 0:
            return d

        elif w > 0:
            return d.dag()

    def alpha(self, i: Union[int, range], j: int) -> float:
        """Return the alpha, amount of displacement, for the bosonic
        displacement operator for junction i and mode j.

        Parameters
        ----------
            i:
                Index of the Junction. (starts from zero for the first mode)
            j:
               Index of the mode. (starts from zero for the first mode)
        """

        Z = np.sqrt(self.cInvTrans[j, j] / self.lTrans[j, j])

        coef = 2 * np.pi / unt.Phi0 * 1j

        return coef * np.sqrt(unt.hbar/2*Z) * self.wTrans[i, j]

    def _build_op_memory(self) -> None:
        """build the charge, flux, number, and cross multiplication of charge
        operators and store them in memory related to operators.
        """

        charge_ops: List[Qobj] = []
        flux_ops: List[Qobj] = []
        num_ops: List[Qobj] = []
        charge_by_charge_ops: List[List[Qobj]] = []

        for i in range(self.n):

            Q = []
            charges_row = []
            num = []
            flux = []
            for j in range(self.n):
                if i == j:
                    Q_iso = self._charge_op_isolated(j)
                    Q2 = Q + [Q_iso * Q_iso]
                    # append the rest with qeye.
                    Q2 += [qt.qeye(self.m[k]) for k in range(j+1, self.n)]
                    charges_row.append(self._squeeze_op(qt.tensor(*Q2)))

                    Q.append(Q_iso)
                    num.append(self._num_op_isolated(j))
                    flux.append(self._flux_op_isolated(j))
                else:
                    if j > i:
                        QQ = Q + [self._charge_op_isolated(j)]
                        # append the rest with qeye.
                        QQ += [qt.qeye(self.m[k]) for k in range(j+1, self.n)]
                        charges_row.append(self._squeeze_op(qt.tensor(*QQ)))

                    Q.append(qt.qeye(self.m[j]))
                    num.append(qt.qeye(self.m[j]))
                    flux.append(qt.qeye(self.m[j]))

            charge_ops.append(self._squeeze_op(qt.tensor(*Q)))
            num_ops.append(self._squeeze_op(qt.tensor(*num)))
            flux_ops.append(self._squeeze_op(qt.tensor(*flux)))
            charge_by_charge_ops.append(charges_row)

        self._memory_ops["Q"] = charge_ops
        self._memory_ops["QQ"] = charge_by_charge_ops
        self._memory_ops["phi"] = flux_ops
        self._memory_ops["N"] = num_ops

    def _build_exp_ops(self) -> None:
        """Build exponential operators needed to construct cosine potential of
        the Josephson Junctions and store them in memory related to operators.
        Note that cosine of JJs can be written as summation of two
        exponential terms,cos(x)=(exp(ix)+exp(-ix))/2. This function builds
        the quantum operators for only one exponential terms.
        """

        # list of exp operators
        exp_ops = []
        # list of square root of exp operators
        root_exp_ops = []

        # number of Josephson Junctions
        nJ = self.wTrans.shape[0]

        for i in range(nJ):

            # list of isolated exp operators
            exp = []
            # list of isolated square root of exp operators
            exp_h = []

            # tensor multiplication of displacement operator for JJ Hamiltonian
            for j in range(self.n):

                if self._is_charge_mode(j):
                    exp.append(self._d_op_isolated(j, self.wTrans[i, j]))
                    exp_h.append(qt.qeye(self.m[j]))
                else:
                    exp.append(qt.displace(self.m[j], self.alpha(i, j)))
                    exp_h.append(qt.displace(self.m[j], self.alpha(i, j)/2))

            exp_ops.append(self._squeeze_op(qt.tensor(*exp)))
            root_exp_ops.append(self._squeeze_op(qt.tensor(*exp_h)))

        self._memory_ops["exp"] = exp_ops
        self._memory_ops["root_exp"] = root_exp_ops

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
                    if self._is_charge_mode(i):
                        LC_hamil += (0.5 * self.cInvTrans[i, i]
                                     * self._memory_ops["QQ"][i][j])
                    else:
                        LC_hamil += self.omega[i] * self._memory_ops["N"][i]

                elif j > 0:
                    if self.cInvTrans[i, i + j] != 0:
                        LC_hamil += (self.cInvTrans[i, i + j]
                                     * self._memory_ops["QQ"][i][j])

        return LC_hamil

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
            op = self.coupling_op("inductive", edge)
            H += x * phi * (unt.Phi0 / 2 / np.pi) * op / np.sqrt(unt.hbar)

            # save the operators for loss calculation
            self._memory_ops["ind_hamil"][(el, B_idx)] = op

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

    def charge_op(self, mode: int, basis: str = 'FC') -> Qobj:
        """Return charge operator for specific mode in the Fock/Charge basis or
        the eigenbasis.

        Parameters
        ----------
            mode:
                Integer that specifies the mode number.
            basis:
                String that specifies the basis. It can be either ``"FC"``
                for original Fock/Charge basis or ``"eig"`` for eigenbasis.
        """

        error1 = "Please specify the truncation number for each mode."
        assert len(self.m) != 0, error1

        # charge operator in Fock/Charge basis
        Q_FC = self._memory_ops["Q"][mode-1]

        if basis == "FC":

            return Q_FC

        elif basis == "eig":

            # number of eigenvalues
            n_eig = len(self.efreqs)

            Q_eig = np.zeros((n_eig, n_eig), dtype=complex)

            for i in range(n_eig):
                for j in range(n_eig):
                    Q_eig[i, j] = (self._evecs[i].dag()
                                   * Q_FC * self._evecs[j]).data[0, 0]

            return qt.Qobj(Q_eig)

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
                ndarray of eigenfrequencies in frequency unit of SQcircuit
                (gigahertz by default)
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
                if self._is_charge_mode(mode):
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
                if self._is_charge_mode(i):
                    for j in range(self.n):
                        op += (self.cInvTrans[i, j] * self._memory_ops["Q"][j]
                               / np.sqrt(unt.hbar))
                    partial_omega = np.abs((state2.dag()*op*state2 -
                                            state1.dag()*op*state1).data[0, 0])
                    A = (self.charge_islands[i].A * 2 * unt.e)
                    decay += self._dephasing(A, partial_omega)

        elif dec_type == "cc":
            for el, B_idx in self._memory_ops['cos']:
                partial_omega = self._get_partial_omega_mn(el, states=states,
                                                           _B_idx=B_idx)
                A = el.A * el.value(self.random)
                decay += self._dephasing(A, partial_omega)

        elif dec_type == "flux":
            for loop in self.loops:
                partial_omega = self._get_partial_omega_mn(loop, states=states)
                A = loop.A
                decay += self._dephasing(A, partial_omega)

        return decay

    def _get_quadratic_Q(self, A: ndarray) -> Qobj:
        """Return quadratic form of 1/2 * Q^T * A * Q

        Parameters
        ----------
            A:
                ndarray matrix that specifies the coefficient for
                quadratic expression.
        """

        op = qt.Qobj()

        for i in range(self.n):
            for j in range(self.n-i):
                if j == 0:
                    op += 0.5 * A[i, i+j] * self._memory_ops["QQ"][i][j]
                elif j > 0:
                    op += A[i, i+j] * self._memory_ops["QQ"][i][j]

        return op

    def _get_quadratic_phi(self, A: ndarray) -> Qobj:
        """Get quadratic form of 1/2 * phi^T * A * phi

        Parameters
        ----------
            A:
                ndarray matrix that specifies the coefficient for
                quadratic expression.
        """

        op = qt.Qobj()

        # number of harmonic modes
        n_H = len(self.omega != 0)

        for i in range(n_H):
            for j in range(n_H):
                phi_i = self._memory_ops["phi"][i].copy()
                phi_j = self._memory_ops["phi"][j].copy()
                if i == j:
                    op += 0.5 * A[i, i] * phi_i ** 2
                elif j > i:
                    op += A[i, j] * phi_i * phi_j

        return op

    def _get_partial_H(
            self,
            el: Union[Capacitor, Inductor, Junction, Loop],
            _B_idx: Optional[int] = None,
    ) -> Qobj:
        """
        return the gradient of the Hamiltonian with respect to elements or
        loop as ``qutip.Qobj`` format.

        Parameters
        ----------
            el:
                Element of a circuit that can be either ``Capacitor``,
                ``Inductor``, ``Junction``, or ``Loop``.
            _B_idx:
                Optional integer point to each row of B matrix (external flux
                distribution of that element). This uses to specify that
                gradient is calculated based on which JJ of the circuit
                specifically (we use this option for critical current noise
                calculation)
        """

        partial_H = qt.Qobj()

        if isinstance(el, Capacitor):

            cInv = np.linalg.inv(self.C)
            A = -self.R.T @ cInv @ self.partial_C[el] @ cInv @ self.R
            partial_H += self._get_quadratic_Q(A)

        elif isinstance(el, Inductor):

            A = -self.S.T @ self.partial_L[el]  @ self.S
            partial_H += self._get_quadratic_phi(A)

            for edge, el_ind, B_idx in self.inductor_keys:
                if el == el_ind:

                    phi = self._get_external_flux_at_element(B_idx)

                    partial_H += -(self._memory_ops["ind_hamil"][(el, B_idx)]
                                   / el.value()**2 / np.sqrt(unt.hbar)
                                   * (unt.Phi0/2/np.pi) * phi)

        elif isinstance(el, Loop):

            loop_idx = self.loops.index(el)

            for edge, el_ind, B_idx in self.inductor_keys:
                partial_H += (self.B[B_idx, loop_idx]
                              * self._memory_ops["ind_hamil"][(el_ind, B_idx)]
                              / el_ind.value() * unt.Phi0 / np.sqrt(unt.hbar)
                              / 2 / np.pi)

            for edge, el_JJ, B_idx, W_idx in self.junction_keys:
                partial_H += (self.B[B_idx, loop_idx] * el_JJ.value()
                              * self._memory_ops['sin'][(el_JJ, B_idx)])

        elif isinstance(el, Junction):

            for _, el_JJ, B_idx, W_idx in self.junction_keys:

                if el == el_JJ and _B_idx is None:
                    partial_H += -self._memory_ops['cos'][(el, B_idx)]

                elif el == el_JJ and _B_idx == B_idx:
                    partial_H += -self._memory_ops['cos'][(el, B_idx)]

        return partial_H

    def _get_partial_omega(
            self,
            el: Union[Capacitor, Inductor, Junction, Loop],
            m: int,
            subtract_ground: bool = True,
            _B_idx: Optional[int] = None,
    ) -> float:
        """Return the gradient of the eigen angular frequency with respect to
        elements or loop as ``qutip.Qobj`` format.

        Parameters
        ----------
            el:
                Element of a circuit that can be either ``Capacitor``,
                ``Inductor``, ``Junction``, or ``Loop``.
            m:
                Integer specifies the eigenvalue. for example ``m=0`` specifies
                the ground state and ``m=1`` specifies the first excited state.
            _B_idx:
                Optional integer point to each row of B matrix (external flux
                distribution of that element). This uses to specify that
                gradient is calculated based on which JJ of the circuit
                specifically (we use this option for critical current noise
                calculation)

        """

        state_m = self._evecs[m]

        partial_H = self._get_partial_H(el, _B_idx)

        partial_omega_m = state_m.dag() * (partial_H*state_m)

        if subtract_ground:

            state_0 = self._evecs[0]

            partial_omega_0 = state_0.dag() * (partial_H * state_0)

            return (partial_omega_m - partial_omega_0).data[0, 0].real

        else:

            return partial_omega_m.data[0, 0].real

    def _get_partial_omega_mn(
            self,
            el: Union[Capacitor, Inductor, Junction, Loop],
            states: Tuple[int, int],
            _B_idx: Optional[int] = None,
    ) -> float:
        """Return the gradient of the eigen angular frequency with respect to
        elements or loop as ``qutip.Qobj`` format. Note that if
        ``states=(m, n)``, it returns ``partial_omega_m - partial_omega_n``.

        Parameters
        ----------
            el:
                Element of a circuit that can be either ``Capacitor``,
                ``Inductor``, ``Junction``, or ``Loop``.
            m:
                Integer specifies the eigenvalue. for example ``m=0`` specifies
                the ground state and ``m=1`` specifies the first excited state.
        """

        state_m = self._evecs[states[0]]
        state_n = self._evecs[states[1]]

        partial_H = self._get_partial_H(el, _B_idx)

        partial_omega_m = state_m.dag() * (partial_H*state_m)
        partial_omega_n = state_n.dag() * (partial_H*state_n)

        return (partial_omega_m - partial_omega_n).data[0, 0].real

    def _get_partial_vec(self, el, m):
        """Return the gradient of the eigenvectors with respect to
        elements or loop as ``qutip.Qobj`` format.

        Parameters
        ----------
            el:
                Element of a circuit that can be either ``Capacitor``,
                ``Inductor``, ``Junction``, or ``Loop``.
            m:
                Integer specifies the eigenvalue. for example ``m=0`` specifies
                the ground state and ``m=1`` specifies the first excited state.
        """

        state_m = self._evecs[m]

        #     state_m = state_m*np.exp(-1j*np.angle(state_m[0,0]))

        n_eig = len(self._evecs)

        partial_H = self._get_partial_H(el)
        partial_state = qt.Qobj()

        for n in range(n_eig):

            if n == m:
                continue

            state_n = self._evecs[n]

            delta_omega = (self._efreqs[m] - self._efreqs[n])

            partial_state += (state_n.dag()
                              * (partial_H * state_m)) * state_n / delta_omega

        return partial_state
