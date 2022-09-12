#!/bin/bash -x
mpisubmit.pl -p $1 -w 00:15 --stdout neyman_pde_mpi.$(jobid).$1.$2_$3.out ../neyman_pde_mpi $2 $3
