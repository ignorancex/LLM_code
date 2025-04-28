"""
Contains different methods for calculating the relevant energies of a Duffing chain with a ZZ interaction.

"""

from itertools import product

import numpy as np
import qutip as qt

from qcheff.duffing.duffing_chain_utils import (
    create_linear_spectrum_zz_chain,
    duffing_chain_num_ham,
    duffing_chain_qutip_ham,
    duffing_chain_scq_hilbertspace,
)
from qcheff.npad.sparse.npad_cpu import npad_diagonalize, npad_diagonalize_subspace


def scq_auto_find_evals(level_labels, del1, del2, wr=5, nr=1, debug=False):
    """
    Return the desired energies of a Duffing chain using scqubits.
    The dressed states are calculated numerically and mapped to the bare states using scqubits automatically.
    """
    test_hs = duffing_chain_scq_hilbertspace(
        create_linear_spectrum_zz_chain(
            delta1=del1,
            delta2=del2,
            omega_res=wr,
            num_resonators=nr,
        )
    )
    # end_zz_bare_labels = [tuple(l + ([0] * nr) + r) for l, r in product([[0], [1]], repeat=2)]
    evals_in_order = list(map(test_hs.energy_by_bare_index, level_labels))
    return evals_in_order


def scq_overlap_find_evals(level_labels, del1, del2, wr=5, nr=1, debug=False):
    """
    Return the energies of the four lowest energy states of a Duffing chain with a ZZ interaction using the scqubits.
    The dressed states are calculated numerically and mapped to the bare states using an overlap matrix manually.
    """
    test_hs = duffing_chain_scq_hilbertspace(
        create_linear_spectrum_zz_chain(
            delta1=del1,
            delta2=del2,
            omega_res=wr,
            num_resonators=nr,
        )
    )
    test_ham = test_hs.hamiltonian()
    levels_states = list(map(test_hs.bare_productstate, level_labels))
    test_evals, test_esys = qt.Qobj(test_ham).eigenstates()
    overlap_mat = np.asarray(
        [
            np.abs(test_estate.overlap(state)) ** 2
            for state, test_estate in product(levels_states, test_esys)
        ]
    ).reshape(4, -1)
    evals_in_order = test_evals[np.argmax(overlap_mat, axis=1)]
    return evals_in_order


def qutip_overlap_find_evals(level_labels, del1, del2, wr=5, nr=1, debug=False):
    """
    Return the energies of the four lowest energy states of a Duffing chain with a ZZ interaction using qutip.
    The dressed states are calculated numerically and mapped to the bare states using an overlap matrix manually.
    """
    test_ham = duffing_chain_qutip_ham(
        create_linear_spectrum_zz_chain(
            delta1=del1,
            delta2=del2,
            omega_res=wr,
            num_resonators=nr,
        )
    )
    system_dims = [3] * (nr + 2)
    levels_states = [qt.basis(system_dims, list(level)) for level in level_labels]
    test_evals, test_esys = qt.Qobj(test_ham).eigenstates()
    overlap_mat = np.asarray(
        [
            np.abs(test_estate.overlap(state)) ** 2
            for state, test_estate in product(levels_states, test_esys)
        ]
    ).reshape(4, -1)
    evals_in_order = test_evals[np.argmax(overlap_mat, axis=1)]
    return evals_in_order


def npad_auto_find_evals(level_labels, del1, del2, wr=5, nr=1, eps=1e-12, debug=False):
    """
    Return the energies of the four lowest energy states of a Duffing chain with a ZZ interaction using NPAD.
    NPAD preserves state labels, so we don't need to compute overlaps.
    """
    test_ham = duffing_chain_num_ham(
        create_linear_spectrum_zz_chain(
            delta1=del1,
            delta2=del2,
            omega_res=wr,
            num_resonators=nr,
        )
    )
    system_dims = [3] * (nr + 2)
    end_zz_idx = np.asarray(
        [qt.state_number_index(system_dims, state) for state in level_labels]
    )
    npad_cz_ham = npad_diagonalize(
        npad_diagonalize_subspace(
            test_ham,
            subspace_idx=end_zz_idx,
            eps=eps,
        ),
        eps=eps,
    )
    evals_in_order = np.real(npad_cz_ham.diagonal())
    return evals_in_order


def npad_diag_find_evals(level_labels, del1, del2, wr=5, nr=1, eps=1e-12, debug=False):
    """
    Return the energies of the four lowest energy states of a Duffing chain with a ZZ interaction using NPAD.
    NPAD preserves state labels, so we don't need to compute overlaps.
    """
    test_ham = duffing_chain_num_ham(
        create_linear_spectrum_zz_chain(
            delta1=del1,
            delta2=del2,
            omega_res=wr,
            num_resonators=nr,
        )
    )
    system_dims = [3] * (nr + 2)

    end_zz_idx = np.asarray(
        [qt.state_number_index(system_dims, state) for state in level_labels]
    )
    npad_cz_ham = npad_diagonalize(
        test_ham,
        eps=eps,
    )[np.ix_(end_zz_idx, end_zz_idx)]
    evals_in_order = np.real(npad_cz_ham.diagonal())
    return evals_in_order
