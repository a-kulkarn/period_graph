#!bash
# Moves output files from the output directories to the archive.

dirname=../archive/$(tr ' ' '_' <<< $(date))
mkdir $dirname

if {
    cp -r DifferentiateCohomology-failed $dirname/. &&
	cp -r edge-data $dirname/. &&
	cp -r failed-edges $dirname/. &&
	cp -r output-files $dirname/. &&
	cp -r root-quartics $dirname/. &&
	cp -r vertex-data $dirname/. &&
	cp -r ode-data $dirname/. &&
	cp -r periods $dirname/. &&
	((! test -f Z-integration-return-codes.sobj) || cp Z-integration-return-codes.sobj $dirname/.) &&
	((! test -f edge-manifest) || cp edge-manifest $dirname/.)
}; then
    echo "Copy to archive completed."
    # If everything copied OK, delete the old files.
    source .clean-data.sh
fi
