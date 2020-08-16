import numpy as np

def viterbi(sequence, obs2i, p_start=None, p_trans=None, p_stop=None, p_emiss=None):
    """
    Compute the Viterbi sequence from log-probabilities

    Return:
      - best_score (float) the log-probability of the best path
      - best_path (int list) the best path as a list of state IDs
    """

    length = len(sequence)
    num_states = len(p_start)

    # trellis to store Viterbi scores
    trellis = np.full([length, num_states], -np.inf)

    # backpointers to backtrack (to remember what prev. state caused the maximum score)
    # we initialize with -1 values, to represent a non-existing index
    backpointers = -np.ones([length, num_states], dtype=int)

    for i in range(0, num_states):
        trellis[0, i] = p_start[i] + p_emiss[obs2i[sequence[0]], i]

    for j in range(1, length):
        for k in range(num_states):
            prob = trellis[j - 1] + p_trans[k, :] + p_emiss[obs2i[sequence[j]], k]
            trellis[j, k] = np.max(prob)
            backpointers[j, k] = np.argmax(prob)
    trellis[length - 1] += p_stop

    path = [0] * length
    path[length - 1] = np.argmax(trellis[length - 1])

    for i in range(length - 2, -1, -1):
        path[i] = backpointers[i + 1, path[i + 1]]

    best_score = np.max(trellis[length - 1])
    best_path = path
    return best_score, best_path
