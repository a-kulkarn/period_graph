# period_graph
Saves and parallelizes computations of periods of (quartic) hypersurfaces

## Prerequisites
- Magma v2.24 or later
- Sage 9.1
- PeriodSuite: https://github.com/emresertoz/PeriodSuite

## Installation
1) Clone the repository
2) Navigate to the repository root directory
3) Run `./make`

## Associated article
This software originated as an accompaniment to the article `Deep Learning Gauss-Manin Connections`. It can be found at `https://arxiv.org/abs/2007.13786`. The version of the software on the release date of the article is available on the `article` branch.

We have made the associated dataset available [here](https://www.dropbox.com/sh/a9dq3wa6dr61ahf/AADfn1L1QYZ5-ajDUrxISsnBa?dl=0)

Its contents are:
- A magma readable list of tuples, consisting of
  - quartics,
  - the computed precision on their periods,
  - the Picard number,
  - the Picard lattice,
  - the coordinates of the hyperplane section,
  - the endomorphism algebra, and
  - whether the K3 surface has real multiplication.

- A compressed archive of all of the periods we have computed, as sage objects.

- A compressed archive of all of the periods we have computed, readable by magma (using PeriodSuite's loadPeriods function).

## Basic useage
The main exported functionality of this package are the functions

Function     | Description
------------ | -------------
ivps(E)              | Constructs the Picard-Fuchs ODEs associated to the pencils of quartic surfaces specified by E.
first_ivps(E)        | Constructs only the first Picard-Fuchs ODE associated to the pencils of quartic surfaces specified by E.
integrate_odes(E)    | Solves the initial value problems set up by the previous function.
nn_sort              | Uses a neural network to anticipate the difficulty of running ivps, and sorts elements of E according to preceived feasibility.
create_training_data | 
train_AI             | Attempts to train a neural network using the data generated in the previous step. Training options are specified by the file `NNCONFIG.py`

# FAQ

1. I got this error during installation:
```
Magma initialization of Fermat. Begin...
Loading "magma/MAGMA_CONFIG"
WriteToFileHomAndCohOfFermat(
    n: 2,
    d: 4
)
PeriodSuiteStorageDir(
)
PathToSuite(
)
In file "/home/akulkarn/pg-install/src/suite/suite.mag", line 18, column 8:
>>   dir:=pathToSuite;
          ^
Runtime error: Package "/home/akulkarn/pg-install/src/suite/pathToSuite.mag" has
not been attached
```
what does it mean?

**Answer:** This is because you are using Magma v23 or less. There is a work-around. Introduce the following line in your `.magmarc` file:
```
Attach("<repo-directory-location>/src/suite/pathToSuite.mag");
```

2. What do I do if something goes wrong?

**Answer:** Most of the time, something is wrong in the config files. These are generated on your system, so you may want to double check that the paths are correct. The repo is pretty young, so it's also often the case that it's just a bug somewhere in the source code. Please send the error to us, and we'll try our best to contact you about a fix.

3. I tried to use `ivps(E)` or `integrate_edge_odes(E)` and nothing happened.

**Answer:**  Presently, there is a caching system in place to prevent repeated work from occuring. If you ran a similar job before, it's likely that this is what you are experiencing. We are working on having multiple saved period graphs at once.

4. Can I use the latest branch of *PeriodSuite* after installing this package?

Maybe, but probably yes. Since *PeriodSuite* is a git submodule, you can checkout the latest branch from the origin repository. You'll have to reinitialize the submodules manually if you want to install to factory settings.

