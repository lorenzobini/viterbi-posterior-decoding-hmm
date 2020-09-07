# Viterbi and Posterior Decoding path prediction on HMM

This repository contains a Python implementation of the Viterbi algorithm and the Posterior Decoding
methods for state prediction given a Hidden Markov Model and a set
of known observations. 

## Getting Started

To execute the algorithms it is necessary to provide a training and a test set within
two distinct .txt files stored within the /data folder. \
The file train.txt contains the training set. The data has to be structured as follows:

* observation/state 
* Multiple observations/state of a path within the HMM must be separated by a space
* Multiple paths withing the HMM must be placed on different lines 

The file test.txt contains the test observations. The data has to be structured as follows:

* Multiple observations of a path within the HMM must be separated by a space
* Multiple observation paths within the HMM must be placed on different lines

A sample of both the training and the test set is provided. \
Run the main function to generate the state prediction for each of the test observations
using both Viterbi and Posterior Decoding. 

### Prerequisites

Install the following prerequisites:

[packages]
* numpy

[requires]
* python_version = "3.7"

## Authors

* **Lorenzo Bini**
* **Mike Nies**

## Credits
Fragments of the code used is courtesy of **T. Deoskar**.

## Future work
* Generation of extensive train and test paths given the specification of an Hidden Markov Model. 
* Output refinement