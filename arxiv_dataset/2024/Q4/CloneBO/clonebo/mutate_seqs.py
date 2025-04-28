import numpy as np

def get_mutations(input_seqs, only_subs=False):
    """For each seq, return [len_seq, alphabet_size] subs,
    [len_seq+1, alphabet_size] insertions, [len_seq] deletions and concatenate them.
    Turn invalid muts into nans.
    
    Parameters:
    input_seqs: numpy array
        Sequences in OHE representation. Can have any number of dims. Must not have
        a stop. Assumes last entry of OHE representation is empty (raises error if not).
    only_subs: bool or int, default = False
        Only return substitutions. If int, only return substitutions within only_subs
        in the cyclic nbh.

    Returns:
    all_seqs: numpy array
        Same OHE length as original, so if not only_subs, make sure input seqs has extra
        OHE column (raises warning if not).
    signs: numpy array
        Signs from lexicographic ordering: 1 for deletions, -1 for insertions,
        1 if substitution is > wild-type and -1 otherwise.
    """
    assert np.sum(input_seqs[..., -1, :]) == 0, "No extra entry in OHE!"
    shape = np.shape(input_seqs)[:-2]
    seqs = input_seqs.reshape((-1,) + np.shape(input_seqs)[-2:]).astype(float)
    num_seqs, seq_len, alphabet_size = np.shape(input_seqs)
    assert alphabet_size > 1 or not only_subs, "no subs for single letter alphabet!"
    seq_len = seq_len - 1
    empty = np.sum(seqs, axis=-1) == 0 # empty positions
    empty_for_ins = np.concatenate(
        [np.zeros([num_seqs, 1], dtype=bool), empty[:, :-1]], axis=-1)

    # First substitutions.
    if alphabet_size > 1:
        # take only nbh if only_subs is an int
        if not isinstance(only_subs, bool):
            nbh = np.r_[np.arange(only_subs),
                        alphabet_size - 2 - np.arange(only_subs)]
            nbh = np.isin(range(alphabet_size-1), nbh)
        else:
            nbh = np.s_[:]
        perm_mat = np.eye(alphabet_size)
        perm_mat = np.r_[perm_mat[1:], perm_mat[[0]]]
        substitutions = np.array([[
            np.concatenate([seqs[:, :i, :],
                            np.einsum('nb,bk->nk', seqs[:, i],
                                      np.linalg.matrix_power(perm_mat, b+1))[:, None, :],
                            seqs[:, i+1:, :]], axis=-2)
            for b in range(alphabet_size-1)] for i in range(seq_len)])
        substitutions = np.transpose(substitutions, [2, 0, 1, 3, 4])[:, :, nbh, :, :]
        substitutions[empty[:, :-1]] = np.nan # substitutions in empty positions are nan'ed
        substitutions=substitutions.reshape([num_seqs, -1, seq_len+1, alphabet_size])
        sub_signs = np.cumsum(seqs[..., ::-1], axis=-1)[..., ::-1]
        sub_signs = sub_signs[..., 1:][..., ::-1]
        sub_signs[np.tile(empty[..., None], len(np.shape(empty))*(1,)
                          +(np.shape(sub_signs)[-1],))] = np.nan
        sub_signs = sub_signs[..., :-1, :][..., nbh]
        sub_signs = 2 * sub_signs.reshape([num_seqs, -1]) - 1
    else:
        substitutions = []
        sub_signs = []

    if not only_subs:
        # Then deletions.
        deletions = np.array([
            np.concatenate([seqs[:, :i, :], seqs[:, i+1:, :],
                            np.tile(np.zeros(alphabet_size)[None, None, :],
                                    (num_seqs, 1, 1))], axis=-2)
            for i in range(seq_len)])    
        deletions = np.transpose(deletions, [1, 0, 2, 3])
        deletions[empty[:, :-1]] = np.nan
        del_signs = np.ones(np.shape(deletions)[:-2])
        del_signs[empty[:, :-1]] = np.nan

        # Finally, insertions
        insertions = np.array([[
            np.concatenate([seqs[:, :i, :],
                            np.tile(base[None, None, :], (num_seqs, 1, 1)),
                            seqs[:, i:-1, :]], axis=-2)
            for base in np.eye(alphabet_size)] for i in range(seq_len+1)])
        insertions = np.transpose(insertions, [2, 0, 1, 3, 4])
        insertions[empty_for_ins] = np.nan
        ins_signs = - np.ones(np.shape(insertions)[:-2])
        ins_signs[empty_for_ins] = np.nan
        insertions = insertions.reshape([num_seqs, -1, seq_len+1, alphabet_size])
        ins_signs = ins_signs.reshape([num_seqs, -1])
    
    # Concatenate.
    if only_subs:
        all_seqs = substitutions
        all_signs = sub_signs
    else:
        all_seqs = np.concatenate([substitutions]*(alphabet_size>1)
                                  + [insertions, deletions], axis=1).reshape(
            shape + (-1,) + np.shape(input_seqs)[-2:])
        all_signs = np.concatenate([sub_signs]*(alphabet_size>1)
                                   + [ins_signs, del_signs], axis=1)
    return all_seqs, all_signs