

// IMPLICIT INPUTS TO THE SCRIPT:
//    name   -- identifier to avoid parallel file access.
//    f0_vec -- coefficients of first polynomial, as an integer list.
//    f1_vec -- coefficients of second polynomial, as an integer list.
//    bias   -- Should be set to "None" in this script.

load "magma/MAGMA_CONFIG";
Attach(SUITE_FILE);

P<x,y,z,w> := PolynomialRing(Rationals(), 4);
mons   := MonomialsOfDegree(P,4);

/* f0_vec := eval(f0_vec); */
/* f1_vec := eval(f1_vec); */
/* bias   := eval(bias); */
/* f0 := &+[ f0_vec[i]*mons[i] : i in [1..#mons]]; */
/* f1 := &+[ f1_vec[i]*mons[i] : i in [1..#mons]]; */

//f0 := eval(f0);
//f1 := eval(f1);

/* Just set a bulk alarm time. In theory, the best thing is to have timeouts during any
   individual call, via making this a parameter in PicardFuchs (as well as every other functions).
   Unfortunately, it is philosophically undesirable to have a function call
   terminate an entire magma session. If only magma could handle timeouts reasonably... */
// Alarm(21*30);

function S4_transition(A_as_list, fstr)
    f := eval(fstr);
    A := Matrix(Integers(), 4, 4, eval(A_as_list));
    rows := Rows(translate_period_matrix(A, f));
    return [Eltseq(r) : r in rows];
end function;
