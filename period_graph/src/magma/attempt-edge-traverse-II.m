
// IMPLICIT INPUTS TO THE SCRIPT:
//    suffix -- identifier to avoid parallel file access.
//    f0_vec -- coefficients of first polynomial, as an integer list.
//    f1_vec -- coefficients of second polynomial, as an integer list.
//    bias   -- Either the string "None", or a number indicating the maximum
//              entropy allowed for the output of DifferentiateCohomology.

SetQuitOnError(true);
SetColumns(0);
load "magma/MAGMA_CONFIG";
Attach(SUITE_FILE);
load "magma/create-nn-data-II.m";

//*************************** 
// START MAIN COMPUTATION
//

SetVerbose("AIOutput", true);
SetVerbose("PicardFuchs", false);

Alarm(eval(timeout));

// Picard-Fuchs computation.
t0:=Cputime();

ft, f0, f1 := ParseInput(f0_vec, f1_vec);
ode:=PicardFuchs([1],ft);


t1:=Cputime();
wt:=(t1-t0);

// Extract output data and print to python.
ode_order,ode_degree := Explode(Analytics(ode)[1]);

print "DataLabel:", [wt,ode_order,ode_degree];
exit;

