import numpy as np


def logsum(logv):
    """Sum of probabilities in log-domain."""
    res = -np.inf
    for val in logv:
        res = logsum_pair(res, val)
    return res


def logsum_pair(logx, logy):
    """
    Return log(x+y), avoiding arithmetic underflow/overflow.
    """
    if logx == -np.inf:
        return logy
    elif logx > logy:
        return logx + np.log1p(np.exp(logy - logx))
    else:
        return logy + np.log1p(np.exp(logx - logy))


def forward(sequence, p_start=None, p_trans=None, p_stop=None, p_emiss=None):
    """
    Compute Forward probabilities.
    Note: all probabilities should be log-probabilities.

    Return:
      - trellis with forward probabilities, excluding the "stop" cell
      - the forward probability of the stop cell (this is the log-likelihood!)
    """

    length = len(sequence)
    num_states = len(p_start)

    # trellis to store forward probabilities
    trellis = np.full([length, num_states], -np.inf)
    log_likelihood = np.full(num_states, -np.inf)

    if length == 1:
        # base case
        for state in range(0, num_states):
            trellis[length - 1, state] = logsum_pair(p_start[state], p_emiss[obs2i[sequence[-1]], state])
    else:
        # recursion
        trellis[0:length - 1, :], log_likelihood = forward(sequence[0:length - 1], p_start, p_trans, p_stop, p_emiss)

        for state in range(0, num_states):
            for prev_state in range(0, num_states):
                trellis[length - 1, state] = logsum([p_trans[state, prev_state], trellis[length - 2, prev_state]])
            trellis[length - 1, state] = logsum_pair(trellis[length - 1, state], p_emiss[obs2i[sequence[-1]], state])
            log_likelihood[state] = logsum([log_likelihood[state],
                                            p_stop[state], trellis[length - 2, state]])

    return trellis, log_likelihood


def backward(sequence, p_start=None, p_trans=None, p_stop=None, p_emiss=None):
    """
    Compute Backward probabilities.
    Note: all probabilities should be log-probabilities.

    Return:
      - trellis with backward probabilities, excluding the "start" cell
      - the forward probability of the start cell (this is ALSO the log-likelihood!)
    """

    length = len(sequence)
    num_states = len(p_start)

    # trellis to store forward probabilities
    trellis = np.full([length, num_states], -np.inf)
    log_likelihood = np.full(num_states, -np.inf)

    if length == 1:
        # base case
        for state in range(0, num_states):
            trellis[length - 1, state] = p_stop[state]
    else:
        # recursion
        trellis[1:, :], log_likelihood = backward(sequence[1:], p_start, p_trans, p_stop, p_emiss)
        for state in range(0, num_states):
            for next_state in range(0, num_states):
                trellis[0, state] = logsum([trellis[0, state],
                                            p_trans[next_state, state], trellis[1, next_state],
                                            p_emiss[obs2i[sequence[1]], next_state]])

            log_likelihood[state] = logsum([log_likelihood[state],
                                            p_start[state], trellis[1, state], p_emiss[obs2i[sequence[1]], state]])

    return trellis, log_likelihood


def forward_backward(sequence):
    """
    Compute forward and backward probabilities.
    Return:
    - fw_trellis
    - fw_log_likelihood (the value of the "stop" cell, not part of the trellis)
    - bw_trellis
    - bw_log_likelihood (the value of the "start" cell, not part of the trellis)
    """
    fw_trellis, fw_ll = forward(sequence, p_start=p_start, p_trans=p_trans, p_stop=p_stop, p_emiss=p_emiss)
    bw_trellis, bw_ll = backward(sequence, p_start=p_start, p_trans=p_trans, p_stop=p_stop, p_emiss=p_emiss)
    return fw_trellis, fw_ll, bw_trellis, bw_ll


def posterior_decode(sequence, fw_trellis, bw_trellis, ll, p_trans, p_emiss):
    """
    Return best hidden state sequence according to Posterior decoding
    """

    length = len(sequence)
    num_states = fw_trellis.shape[1]

    # calculate the state posteriors
    state_posteriors = np.zeros([length, num_states])

    for i in range(length):
        fw = fw_trellis[i, :]
        bw = bw_trellis[i, :]
        for k in range(num_states):
            state_posteriors[i, k] = logsum_pair(fw[k], bw[k])

    state_posteriors = np.exp(state_posteriors)

    # the best states are simply the arg max of the state posteriors
    best_sequence = np.argmax(state_posteriors, axis=1)

    return state_posteriors, best_sequence