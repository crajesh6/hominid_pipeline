#!/bin/bash

# rsat_command="rsat matrix-clustering -matrix test test.transfac transfac -o test/test -v 2"
# echo "Environment variables with 'singularity run':"
# singularity run rsat.sif env
# echo
# echo "Environment variables with 'singularity exec':"
# singularity exec rsat.sif env



# echo "Path to rsat binary with 'singularity run':"
# singularity run rsat.sif which rsat
# echo
# echo "Path to rsat binary with 'singularity exec':"
# singularity exec rsat.sif which rsat

# singularity exec rsat.sif ls -l /packages/rsat/bin/rsat


rsat_binary="/packages/rsat/bin/rsat"

echo "Path to rsat binary with 'singularity run':"
singularity run rsat.sif which rsat
echo
echo "Path to rsat binary with 'singularity exec':"
singularity exec rsat.sif $rsat_binary which rsat
