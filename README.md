# period_graph
Saves and parallelizes computations of periods of (quartic) hypersurfaces

## Prerequisites
- Magma v2.24 or later
- Sage 9.1

## Installation
1) Clone the repository
2) Navigate to the repository root directory
3) Run `./make`

## Associated article
This software originated as an accompaniment to the article `article-name`. It can be found at `link`. The version of the software on the release date of the article is available on the `branch-name` branch.

## Basic useage
The main exported functionality of this package are the functions

Function     | Description
------------ | -------------
* ivps(E)              | Constructs the Picard-Fuchs ODEs associated to the pencils of quartic surfaces specified by E.
* first_ivps(E)        | Constructs only the first Picard-Fuchs ODE associated to the pencils of quartic surfaces specified by E.
* integrate_odes(E)    | Solves the initial value problems set up by the previous function.
* nn_sort              | Uses a neural network to anticipate the difficulty of running ivps, and sorts elements of E according to preceived feasibility.
* create_training_data | 
* train_AI             | Attempts to train a neural network using the data generated in the previous step. Training options are specified by the file `NNCONFIG.py`
