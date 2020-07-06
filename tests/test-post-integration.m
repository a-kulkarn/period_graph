
Attach("../src/suite/suite.mag");

// The directory storing the results of the period computation.
dirname := "../src/periods/";

// Some polynomials where the precision survived.
R<x,y,z,w> := PolynomialRing(Rationals(), 4);

test_polynomials := [
x^3*w + x*w^3 + y^4 + y*z^3,
x^3*w + x*w^3 + y^4 + z^4,
x^4 + y^4 + y*z^3 + w^4,
x^4 + y^3*w + y*w^3 + z^4,
x^4 + x*w^3 + y^4 + y*z^3,
x^3*z + y^4 + z^4 + w^4,
x^3*z + y^4 + z^3*w + w^4,
x^3*z + x*z^3 + y^3*w + y*w^3,
//x^3*y + y^3*z + z^4 + w^4,
//x^3*y + y^3*w + z^4 + z*w^3,
x^3*y + x*y^3 + z^3*w + z*w^3,
//x^3*w + y^4 + z^4 + z*w^3,
x^3*w + x*w^3 + y^4 + z^4,
x^3*w + x*w^3 + y^4 + y*z^3,
x^4 + y^4 + z^4 + w^4
];

test_polynomials := [
//x^3*w + x*w^3 + y^4 + y*z^3,
//x^3*w + x*w^3 + y^4 + z^4,
//x^4 + y^4 + y*z^3 + w^4,
//x^4 + y^3*w + y*w^3 + z^4,
//x^4 + x*w^3 + y^4 + y*z^3,
x^3*z + y^4 + z^4 + w^4,
x^4 + x*w^3 + y^4 + z^4,
//x^3*z + y^4 + z^3*w + w^4,
//x^3*z + x*z^3 + y^3*w + y*w^3,
//x^3*y + y^3*z + z^4 + w^4,
//x^3*y + y^3*w + z^4 + z*w^3,
//x^3*y + x*y^3 + z^3*w + z*w^3,
//x^3*w + y^4 + z^4 + z*w^3,
//x^3*w + x*w^3 + y^4 + z^4,
//x^3*w + x*w^3 + y^4 + y*z^3,
x^4 + y^4 + z^4 + w^4
];

test_polynomials := [
x^4 + y^4 + z^4 + w^4,
x^4 + y^4 + z^3*w + z*w^3
];


// Use system to write the directory names to a file.
// Then parse the names to get the quartics.
System("bash -c 'ls ../periods > .test_quartics'");

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


// Conduct the actual test.
test_result := [];

Mfermat := loadPeriods(dirname*"x^4 + y^4 + z^4 + w^4/periods-magma");

for f in test_polynomials do

    if #Monomials(f) ne 4 then
	continue;
    end if;
    
    print "-----------------------------------------------------";
    print "Testing: ", f;
	      
    filename := dirname*Sprint(f)*"/periods-magma";
    M, prec := loadPeriods(filename);

    primitive_rank, basis := LatticeOfRelations(Matrix(Rows(M)[1]), prec);
    pic_rank := primitive_rank+1;
    test_pic_rank := picardNumberOfDelsarte(f);

    test_diff := Abs(test_pic_rank - pic_rank);
    
    test_result cat:= [test_diff];

    print "Polynomial: ", f;
    print "prec: ", prec;

    try
	mat_is_fermat := M eq Mfermat;
    catch e
	mat_is_fermat := M[1] eq Mfermat[1];
    end try;
    
    print "M is fermat matrix?: " , mat_is_fermat;
    print "Result: ", test_diff, "\n";
end for;

print "-----------------------------------------------------";
print "Combinatorial test result: ", test_result;
