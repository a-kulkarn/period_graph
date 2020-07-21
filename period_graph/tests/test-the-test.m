
Attach("../suite/suite.mag");


// Check a trivial case to make sure the test works.

// The directory storing the results of the period computation.
dirname := "../periods/";


// Super basic function to compare periods. Needs some division-by-zero checks.
function ComparePeriodsTest(bundle, M)

    RR := RealField(7);
    pers1 := bundle`primitivePeriods[1];
    pers2 := Rows(M)[1];

    // Check to make sure nothing dumb happens.
    assert Ncols(pers1) eq Ncols(pers2);
    assert Nrows(pers1) eq Nrows(pers2);
    
    a := pers1[2]/pers1[1];
    b := pers2[2]/pers2[1];

    flag := true;
    for i in [1 .. 21] do
	norm_diff := Norm(pers1[i]/pers1[1] - pers2[i]/pers2[1]);
	
	if not norm_diff lt 10^(-50) then
	    flag := false;
	    print i, RR ! norm_diff;
	end if;
    end for;

    return flag;  
end function;
    

// Some polynomials where the precision survived.
R<x,y,z,w> := PolynomialRing(Rationals(), 4);

f0 := x^4 + y^4 + z^4 + w^4;

// The test edges.
test_list := [
    x^4 + y^4 + z^4 + z*w^3,
    x^4 + y^4 + z^3*w + w^4,
    x^4 + y^4 + z^4 + x*w^3,
    x^4 + y^4 + z^4 + y*w^3,
    x^4 + y^4 + x*z^3 + w^4,
    x^4 + x*w^3 + y^4 + z^4]; // The last one checks the permutation edge.


for f1 in test_list do
    M      := loadPeriods(dirname*Sprint(f1)*"/periods-magma");
    bundle := PeriodHomotopy([f0,f1] : precision:=100);

    print "Poly: ", f1;
    error if not ComparePeriodsTest(bundle, M), "Test failed.";
end for;

exit;
