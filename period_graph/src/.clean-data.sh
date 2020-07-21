#!bash
# Removes output files from the output directories.

test "$(ls -A DifferentiateCohomology-failed)" && rm DifferentiateCohomology-failed/*
test "$(ls -A edge-data)" && rm edge-data/*
test "$(ls -A failed-edges)" && rm failed-edges/*
test "$(ls -A output-files)" && rm output-files/*
test "$(ls -A root-quartics)" && rm root-quartics/*
test "$(ls -A vertex-data)" && rm vertex-data/*

test "$(ls -A ode-data)" && rm -r ode-data/*
test "$(ls -A periods)" && rm -r periods/*

test -f Z-integration-return-codes.sobj && rm Z-integration-return-codes.sobj
test -f edge-manifest && rm edge-manifest
