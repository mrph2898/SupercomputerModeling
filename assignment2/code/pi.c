//
// Created by Georgy on 10/31/2021.
//
#include "mpi.h"
#include <stdio.h>

int main (int argc, char *argv[]){
    int n = 100000, myid, numprocs, i;
    double mypi, pi, h, sum, x;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    h = 1.0 / (double) n;
    sum = 0.0;

    for (i= myid+ 1; i<= n; i+= numprocs){
        x = h * ((double)i-0.5);
        sum += (4.0 / (1.0 + x*x));
    }
    mypi= h * sum;
    if (!myid){
        pi= mypi;
        for (i= 1; i<numprocs; i++){
            MPI_Recv(&mypi,1, MPI_DOUBLE, MPI_ANY_SOURCE,MPI_ANY_TAG, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
            pi += mypi;
        }
        printf("PI is approximately %.16f", pi);
    } else
        MPI_Send(&mypi, 1, MPI_DOUBLE,0, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
