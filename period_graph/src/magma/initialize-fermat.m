SetQuitOnError(true);
print "Magma initialization of Fermat. Begin...";
load "magma/MAGMA_CONFIG";
Attach(SUITE_FILE);

// These can be given as user inputs at some point.
n := 2;
d := 4;
phamB,intmat:=WriteToFileHomAndCohOfFermat(n,d);
print "Successful initialization of Fermat. Magma terminating.";
exit;
