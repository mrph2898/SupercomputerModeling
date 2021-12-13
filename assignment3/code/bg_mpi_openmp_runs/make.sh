#!/bin/bash -x
mpixlc_r -qsmp=omp -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 ../neyman_pde_mpi_openmp.c -o neyman_pde_mpi_openmp
