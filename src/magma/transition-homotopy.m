SetQuitOnError(true);

// IMPLICIT INPUTS TO THE SCRIPT:
//    name    -- identifier to avoid parallel file access.
//    f0      -- coefficients of first polynomial, as an integer list.
//    f1      -- coefficients of second polynomial, as an integer list.
//    bias    -- Should be set to "None" in this script.
//    timeout -- parameter controlling timeout.
// only_first -- parameter controlling whether to set the pole order bound.

load "magma/MAGMA_CONFIG";
Attach(SUITE_FILE);
SetVerbose("AIOutput", true);
SetVerbose("PicardFuchs", 2);

P<x,y,z,w> := PolynomialRing(Rationals(), 4);
mons   := MonomialsOfDegree(P,4);

/* f0_vec := eval(f0_vec); */
/* f1_vec := eval(f1_vec); */
/* bias   := eval(bias); */
/* f0 := &+[ f0_vec[i]*mons[i] : i in [1..#mons]]; */
/* f1 := &+[ f1_vec[i]*mons[i] : i in [1..#mons]]; */

f0 := eval(f0);
f1 := eval(f1);

if only_first eq "True" then
    bop := true;
else
    bop := false;
end if;

/* f := Fermat(2,4); */
/* R<x,y,z,w> := Parent(f); */

/* h := f - w^2*x^2; */
/* g := f + x*y*z^2; */


/* Just set a bulk alarm time. In theory, the best thing is to have timeouts during any
   individual call, via making this a parameter in PicardFuchs (as well as every other functions).
   Unfortunately, it is philosophically undesirable to have a function call
   terminate an entire magma session. If only magma could handle timeouts reasonably... */
Alarm(eval(timeout));

// Change the path to output in the current directory.
dirname := SRC_ABS_PATH * "ode-data/";


// Setting integration to false is critical for parallelization.
bundle := TransitionHomotopy([f0,f1] : precision:=100, integrate:=false,
				       name:= dirname*name, overwrite:=true,
				       bound_pole_order:=bop);


// Rely on a separate sage process (possibly on a different server) to do integration.
// Manifest of names to consider needs to be communicated across the network. 

exit;
