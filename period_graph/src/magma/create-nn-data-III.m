
// IMPLICIT INPUTS TO THE SCRIPT:
//    suffix -- identifier to avoid parallel file access.
//    f0_vec -- coefficients of first polynomial, as an integer list.
//    f1_vec -- coefficients of second polynomial, as an integer list.
//    bias   -- Either the string "None", or a number indicating the maximum
//              entropy allowed for the output of DifferentiateCohomology.

SetQuitOnError(true);
SetColumns(0);
load "magma/create-nn-data-II.m";

// Python will capture the output.
Alarm(30);
print DCMData(suffix, f0_vec, f1_vec, bias);
exit;
