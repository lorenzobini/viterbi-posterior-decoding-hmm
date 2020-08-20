import numpy as np
from collections import defaultdict
from src.utilities.utilities import convert_to_log, normalize_all


class Model:

    state2i: defaultdict
    obs2i: defaultdict
    num_states: int
    num_observations: int
    p_start: np.float
    p_trans: np.float
    p_stop: np.float
    p_emiss: np.float

    def __init__(self):
        # initialising mappings from state/obs to an ID
        self.state2i = defaultdict(lambda: len(self.state2i))
        self.obs2i = defaultdict(lambda: len(self.obs2i))

        # initialising dictionaries for number of states and observations
        self.num_states = 0
        self.num_observations = 0

    def build_model(self, training_set, test_set):

        for seq in training_set:
            for example in seq:
                state_id = self.self.state2i[example.state]
                obs_id = self.self.obs2i[example.obs]

        self.num_states = len(self.state2i)
        self.num_observations = len(self.obs2i)

        counts_start = np.zeros(self.num_states)

        # adding 1 every time a sequence starts with a certain state
        for seq in training_set:
            counts_start[self.state2i[seq[0].state]] += 1.

        total = counts_start.sum()
        self.p_start = counts_start / total
        print('start', '-->', self.p_start)

        # initialising transition matrix
        counts_trans = np.zeros([self.num_states, self.num_states])

        # initialising vector for `num_states` values.
        counts_stop = np.zeros(self.num_states)

        # counting transitions one sequence at a time
        for seq in training_set:
            for i in range(1, len(seq)):
                # convert the states to indexes
                prev_state = self.state2i[seq[i - 1].state]
                current_state = self.state2i[seq[i].state]

                # count
                counts_trans[current_state, prev_state] += 1.

        # count final states
        for seq in training_set:
            state = self.state2i[seq[-1].state]
            counts_stop[state] += 1.

        total_per_state = counts_trans.sum(0) + counts_stop
        print("Total counts per state:\n", total_per_state, "\n")

        # normalizing
        self.p_trans = counts_trans / total_per_state
        print("Transition probabilities:\n", self.p_trans)

        # dividing the values in each corresponding index in the 2 vectors
        self.p_stop = counts_stop / total_per_state
        print("Final probabilities:\n", self.p_stop, "\n")

        # initialising matrix to keep track of emission probabilities
        counts_emiss = np.zeros([self.num_observations, self.num_states])

        # count
        for seq in training_set:
            for obs, state in seq:
                obs = self.obs2i[obs]
                state = self.state2i[state]
                counts_emiss[obs][state] += 1.

        # normalize
        self.p_emiss = counts_emiss / counts_emiss.sum(0)

        # sanity check
        try:
            sanity_check(p_start=self.p_start,
                         p_trans=self.p_trans,
                         p_stop=self.p_stop,
                         p_emiss=self.p_emiss)
            print("All good!")
        except AssertionError as e:
            print("There was a problem: %s" % str(e))

        # normalize with smoothing
        smoothing = 0.1
        self.p_start, self.p_trans, self.p_stop, self.p_emiss = normalize_all(
            counts_start, counts_trans, counts_stop, counts_emiss, smoothing=smoothing)

        # convert to log-probabilities
        self.p_start, self.p_trans, self.p_stop, self.p_emiss = convert_to_log(p_start=self.p_start,
                                                                               p_trans=self.p_trans,
                                                                               p_stop=self.p_stop,
                                                                               p_emiss=self.p_emiss)


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
