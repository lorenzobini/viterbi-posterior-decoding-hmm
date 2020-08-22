import os
from collections import namedtuple
import numpy as np
from src.viterbi import viterbi
from src.posterior_decoding import posterior_decode, forward_backward
from src.model import Model


def main():
    test_set, training_set = load_data()

    model = Model()
    model.build_model(training_set, test_set)


    #########################
    #   VITERBI ALGORITHM   #
    #########################

    print("\n VITERBI PATH PREDICTION ---------------\n")
    for test_sequence in test_set:
        best_score, best_path = viterbi(test_sequence,
                                        model.obs2i,
                                        p_start=model.p_start,
                                        p_trans=model.p_trans,
                                        p_stop=model.p_stop,
                                        p_emiss=model.p_emiss)

        i2state = {v: k for k, v in model.state2i.items()}

        print('Sequence: ' + str(test_sequence))
        print('Predicted best score: ' + str(best_score))
        print('Predicted best path: ' + str([i2state[i] for i in best_path]) + '\n')


    ##########################
    #   POSTERIOR DECODING   #
    ##########################

    print("\n POSTERIOR DECODING PATH PREDICTION ---------------\n")
    for test_sequence in test_set:
        fw_trellis, fw_ll, bw_trellis, bw_ll = forward_backward(test_sequence,
                                                                model.obs2i,
                                                                p_start=model.p_start,
                                                                p_trans=model.p_trans,
                                                                p_stop=model.p_stop,
                                                                p_emiss=model.p_emiss)


        state_posteriors, best_sequence = posterior_decode(test_sequence,
                                                           fw_trellis,
                                                           bw_trellis,
                                                           fw_ll,
                                                           model.p_trans,
                                                           model.p_emiss)

        print('Sequence: ' + str(test_sequence))
        print('State posteriors: \n' + str(state_posteriors))
        print('Predicted best path: ' + str([i2state[i] for i in best_path]) + '\n')



def load_data():

    # paths
    DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
    TRAIN_PATH = DATA_PATH + "train.txt"
    TEST_PATH = DATA_PATH + "test.txt"

    if(
        not(
            os.path.isfile(TRAIN_PATH) and
            os.path.isfile(TEST_PATH)
        )
    ):
        raise ImportError("Training and/or test sets not found.")
    else:
        train_data = open(TRAIN_PATH, "r")
        test_data = open(TEST_PATH, "r")

        # processing training set ---------------------------------

        def train_reader(train_lines):
            for line in train_lines:
                yield [Pair(*pair.split("/")) for pair in line.split()]

        # defining a Observation-State pair class
        Pair = namedtuple("Pair", ["obs", "state"])
        Pair.__repr__ = lambda x: x.obs + "/" + x.state

        training_set = list(train_reader(train_data))

        # processing test set --------------------------------------
        def test_reader(test_lines):
            for line in test_lines:
                yield line.split()

        test_set = list(test_reader(test_data))

        print("Training and test sets imported successfully.")


    return test_set, training_set

main()
