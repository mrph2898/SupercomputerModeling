#!/bin/bash -x
mpisubmit.bg -n $1 -w 00:10:00 --stdout neyman_pde_mpi.$(jobid).$1.$2_$3.out neyman_pde_mpi $2 $3
