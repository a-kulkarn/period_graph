

SetColumns(0);
load "magma/MAGMA_CONFIG";
Attach(SUITE_FILE);

function converter(poly)
  P:=Parent(poly);
  mons:=MonomialsOfDegree(P,Degree(poly));
  return [MonomialCoefficient(poly,m) : m in mons];
end function;

// Function to compute the entropy of the DifferentiateCohomology matrix
function Entropy(M)

    L := Eltseq(M);
    vec := [AbsoluteValue(Numerator(x)) : x in L] cat
	   [AbsoluteValue(Denominator(x)) : x in L];

    max := Maximum(vec);
    
    function term_of_ent(x)
	if x eq 0 then
	    return 0;
	else
	    return -(x/max)*Log(x/max);
	end if;
    end function;
    
    return &+ [ term_of_ent(x) : x in vec];
end function;

function ParseInput(f0_vec, f1_vec)
    K<t>   := FunctionField(Rationals());
    P<[x]> := PolynomialRing(Rationals(), 4);
    mons   := MonomialsOfDegree(P,4);

    f0_vec := eval(f0_vec);
    f1_vec := eval(f1_vec);

    f0 := &+[ f0_vec[i]*mons[i] : i in [1..#mons]];
    f1 := &+[ f1_vec[i]*mons[i] : i in [1..#mons]];
    ft := SetupFamily(f0,f1);
    return ft, f0, f1;
end function;

// IMPLICIT INPUTS TO THE SCRIPT:
//    suffix -- identifier to avoid parallel file access.
//    f0_vec -- coefficients of first polynomial, as an integer list.
//    f1_vec -- coefficients of second polynomial, as an integer list.
//    bias   -- Either the string "None", or a number indicating the maximum
//              entropy allowed for the output of DifferentiateCohomology.
//
function DCMData(suffix, f0_vec, f1_vec, bias)

    ft, f0, f1 := ParseInput(f0_vec, f1_vec);
    bias := eval(bias);
    
    // Magma will exit with non-zero error code if the target surface is singular
    assert not IsSingular(Scheme(Proj(Parent(f1)),f1));

    // Compute the first order approximation.
    // If this step fails, then magma will exit with output '' and alarm error code.
    M01:=DifferentiateCohomology(f0,f1);
    M10:=DifferentiateCohomology(f1,f0);

    // Based on the bias, decide whether to record the sample.
    if bias cmpne "None" and Entropy(M01) gt bias then
	error "Entropy is larger than bias."; // discard the sample without output at python level.
    end if;

    // Output the initial parameters. Python will capture the output
    coefs:=converter(f0) cat converter(f1);
    return Sprintf("%o\n%o\n%o\n", coefs, Eltseq(M01), Eltseq(M10));
end function;

/* Output in matrix form.
for row in RowSequence(M01) do
    print row;
end for;
for row in RowSequence(M10) do
    print row;
end for;
*/
