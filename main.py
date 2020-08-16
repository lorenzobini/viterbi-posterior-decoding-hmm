
import numpy as np
from collections import defaultdict
from collections import namedtuple
from helper import convert_to_log, normalize_all
from viterbi import viterbi
from posterior_decoding import posterior_decode, forward_backward



'''
CREDITS: part of the code used is courtesy of T. Deoskar
'''


def main():
    # read in test data
    test_data = """sleep cry laugh cry
    cry cry laugh sleep"""

    def test_reader(test_lines):
        for line in test_lines.splitlines():
            yield line.split()

    test_set = list(test_reader(test_data))

    # read in train data
    train_data = """laugh/happy cry/bored cry/hungry sleep/happy
    cry/bored laugh/happy cry/happy sleep/bored
    cry/hungry cry/bored sleep/happy"""

    # define a Observation-State pair class
    Pair = namedtuple("Pair", ["obs", "state"])
    Pair.__repr__ = lambda x: x.obs + "/" + x.state

    def train_reader(train_lines):
        for line in train_data.splitlines():
            yield [Pair(*pair.split("/")) for pair in line.split()]

    training_set = list(train_reader(train_data))

    '''
    # print the results
    print("test set (observations):")
    for seq in test_set:
        print(seq)
    print("\ntraining set (observation/state pairs):")
    for seq in training_set:
        print(seq)
    '''


    # create mappings from state/obs to an ID
    state2i = defaultdict(lambda: len(state2i))
    obs2i = defaultdict(lambda: len(obs2i))

    for seq in training_set:
        for example in seq:
            state_id = state2i[example.state]
            obs_id = obs2i[example.obs]

    # we can get the number of states and observations from our dictionaries
    num_states = len(state2i)
    num_observations = len(obs2i)

    # this creates a vector of length `num_states` filled with zeros
    counts_start = np.zeros(num_states)

    # now we count 1 every time a sequence starts with a certain state
    # we look up the index for the state that we want to count using the `state2i` dictionary
    for seq in training_set:
        counts_start[state2i[seq[0].state]] += 1.

    print(counts_start)


    # since p_start is a numpy object, we can call sum on it; easy!
    total = counts_start.sum()

    # normalize: divide each count by the total
    p_start = counts_start / total
    print('start', '-->', p_start)

    # initialising transition matrix
    counts_trans = np.zeros([num_states, num_states])

    # initialising vector for `num_states` values.
    counts_stop = np.zeros(num_states)

    # counting transitions one sequence at a time
    for seq in training_set:
        for i in range(1, len(seq)):
            # convert the states to indexes
            prev_state = state2i[seq[i - 1].state]
            current_state = state2i[seq[i].state]

            # count
            counts_trans[current_state, prev_state] += 1.

    # count final states
    for seq in training_set:
        state = state2i[seq[-1].state]
        counts_stop[state] += 1.

    total_per_state = counts_trans.sum(0) + counts_stop
    print("Total counts per state:\n", total_per_state, "\n")

    # normalizing
    p_trans = counts_trans / total_per_state
    print("Transition probabilities:\n", p_trans)

    # dividing the values in each corresponding index in the 2 vectors
    p_stop = counts_stop / total_per_state
    print("Final probabilities:\n", p_stop, "\n")

    # initialising matrix to keep track of emission probabilities
    counts_emiss = np.zeros([num_observations, num_states])

    # count
    for seq in training_set:
        for obs, state in seq:
            obs = obs2i[obs]
            state = state2i[state]
            counts_emiss[obs][state] += 1.

    # normalize
    p_emiss = counts_emiss / counts_emiss.sum(0)

    # sanity check
    try:
        sanity_check(p_start=p_start, p_trans=p_trans, p_stop=p_stop, p_emiss=p_emiss)
        print("All good!")
    except AssertionError as e:
        print("There was a problem: %s" % str(e))

    # normalize with smoothing
    smoothing = 0.1
    p_start, p_trans, p_stop, p_emiss = normalize_all(
        counts_start, counts_trans, counts_stop, counts_emiss, smoothing=smoothing)

    # convert to log-probabilities
    p_start, p_trans, p_stop, p_emiss = convert_to_log(p_start=p_start, p_trans=p_trans, p_stop=p_stop, p_emiss=p_emiss)


    #################################
    #   TESTING VITERBI ALGORITHM   #
    #################################
    test_sequence = test_set[0]
    best_score, best_path = viterbi(test_sequence, obs2i, p_start=p_start, p_trans=p_trans, p_stop=p_stop, p_emiss=p_emiss)

    print(best_score)
    print(best_path)

    i2state = {v: k for k, v in state2i.items()}
    print([i2state[i] for i in best_path])

    ##################################
    #   TESTING POSTERIOR DECODING   #
    ##################################

    test_sequence = test_set[0]

    fw_trellis, fw_ll, bw_trellis, bw_ll = forward_backward(test_sequence,
                                                            obs2i,
                                                            p_start=p_start,
                                                            p_trans=p_trans,
                                                            p_stop=p_stop,
                                                            p_emiss=p_emiss)

    print(test_sequence)
    print(fw_trellis)
    print(fw_ll)
    print(bw_trellis)
    print(bw_ll)

    state_posteriors, best_sequence = posterior_decode(test_sequence, fw_trellis, bw_trellis, fw_ll, p_trans, p_emiss)

    print(state_posteriors)
    print(best_sequence)
    print([i2state[i] for i in best_path])


def almost_one(p, eps=1e-3):
    return (1. - eps) < p < (1. + eps)


def sanity_check(p_start=None, p_trans=None, p_stop=None, p_emiss=None):
    # p_start sanity check
    assert almost_one(p_start.sum())

    # p_trans and p_stop sanity check
    p_trans_stop_sum = np.vstack((p_trans, p_stop)).sum(0)
    for p in p_trans_stop_sum:
        assert almost_one(p)

    # p_emiss sanity check
    p_emiss_sum = p_emiss.sum(0)
    for p in p_emiss_sum:
        assert almost_one(p)


main()
