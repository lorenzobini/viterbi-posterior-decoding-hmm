import math


def convert_to_log(p_start=None, p_trans=None, p_stop=None, p_emiss=None):
    """
    Convert all probabilities to log-probabilities

    Important: only run this function with normal probabilities as input!
    If you run this twice, things will break.
    """

    # converting p_start
    p_start = [math.log(x) for x in p_start]

    # converting p_trans
    i = 0
    for seq in p_trans:
        seq = [math.log(x) for x in seq]
        p_trans[i] = seq
        i += 1

    # converting p_stop
    p_stop = [math.log(x) for x in p_stop]

    # converting p_emiss
    i = 0
    for seq in p_emiss:
        seq = [math.log(x) for x in seq]
        p_emiss[i] = seq
        i += 1

    return p_start, p_trans, p_stop, p_emiss


def normalize(x, smoothing=0.1, axis=0):
    smoothed = x + smoothing
    return smoothed / smoothed.sum(axis)


def normalize_all(counts_start, counts_trans, counts_stop, counts_emiss, smoothing=0.1):
    """Normalize all counts to probabilities, optionally with smoothing."""
    p_start = normalize(counts_start, smoothing=smoothing)
    p_emiss = normalize(counts_emiss, smoothing=smoothing)

    counts_trans_smoothed = counts_trans + smoothing
    counts_stop_smoothed = counts_stop + smoothing
    total_trans_stop = counts_trans_smoothed.sum(0) + counts_stop_smoothed
    p_trans = counts_trans_smoothed / total_trans_stop
    p_stop = counts_stop_smoothed / total_trans_stop

    return p_start, p_trans, p_stop, p_emiss


