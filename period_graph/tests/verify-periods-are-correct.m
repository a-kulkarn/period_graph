
load "magma-test-lib.m";

test_list, R := PolynomialsWithPeriods();
f0 := RFermat(R);


for f1 in test_list do
    if f1 eq f0 then continue; end if;
    
    print "Poly: ", f1;
    M, prec  := loadPeriods(PeriodsFile(f1));
    bundle_s := CreatePeriodBundle(f1 : precision:=prec, primitivePeriods:=M);
				  
    bundle_m := PeriodHomotopy([f0,f1] : precision:=100);

    error if not ComparePeriodsTest(bundle_m, bundle_s), "Test failed.";
    //ComparePeriodsTest(bundle_m, bundle_s);
end for;

exit;
