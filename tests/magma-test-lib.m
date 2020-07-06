
Attach("../src/suite/suite.mag");

// The directory storing the results of the period computation.
PERIODS_DIRNAME := "../src/periods/";

// Obtain the list of polynomials for which the software has computed the periods.
function PolynomialsWithPeriods()
    
    // Some polynomials where the precision survived.
    R<x,y,z,w> := PolynomialRing(Rationals(), 4);

    // Use system to write the directory names to a file.
    // Then parse the names to get the quartics.
    System("bash -c 'ls "*PERIODS_DIRNAME*"../periods > .test_quartics'");

    poly_file := Open(".test_quartics", "r");
    test_polynomials := [];

    while true do
	str := Gets(poly_file);
	if IsEof(str) then
	    break;
	end if;
	g := eval str;
	test_polynomials cat:= [g];
    end while;

    return test_polynomials, R;
end function;

// Create the fermat polynomial in the parent R.
function RFermat(R)
    return &+[R.i^4 : i in [1..Rank(R)]];
end function;

function PeriodsFile(f)
    return PERIODS_DIRNAME*Sprint(f)*"/periods-magma";
end function;

// Super basic function to compare periods. Needs some division-by-zero checks.
function ComparePeriodsTest(bundle1, bundle2)

    RR := RealField(7);
    pers1 := bundle1`primitivePeriods[1];
    pers2 := bundle2`primitivePeriods[1];

    // Check to make sure nothing dumb happens.
    assert Ncols(pers1) eq Ncols(pers2);
    assert Nrows(pers1) eq Nrows(pers2);

    return iso_of_k3s(bundle1, bundle2);
end function;


/*
function ComparePeriodsTest(bundle, M)

    RR := RealField(7);
    pers1 := bundle`primitivePeriods[1];
    pers2 := Rows(M)[1];

    // Check to make sure nothing dumb happens.
    assert Ncols(pers1) eq Ncols(pers2);
    assert Nrows(pers1) eq Nrows(pers2);

    //flag := isomorphism_of_k3s
        
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
*/
