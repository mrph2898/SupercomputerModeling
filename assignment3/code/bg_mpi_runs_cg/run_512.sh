#!/bin/bash -x
mpisubmit.bg -n $1 -w 00:05:00 --stdout neyman_pde_mpi_cg.$(jobid).$1.$2_$3.out ../neyman_pde_mpi_cg $2 $3
