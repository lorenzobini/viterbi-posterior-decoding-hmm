

from collections import namedtuple

from src.viterbi import viterbi
from src.posterior_decoding import posterior_decode, forward_backward
from src.model import Model




'''
CREDITS: part of the code used is courtesy of T. Deoskar
'''


def main():
    test_set, training_set = load_data()

    model = Model()
    model.build_model(training_set, test_set)



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


def load_data():
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

    # print the results
    print("test set (observations):")
    for seq in test_set:
        print(seq)
    print("\ntraining set (observation/state pairs):")
    for seq in training_set:
        print(seq)

    return test_set, training_set


main()
