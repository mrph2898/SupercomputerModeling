#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>
#include <math.h>


double func(double x, double y, double z){
    if ((x*x + y*y <= z*z)  && ((z >= 0) && (z <= 1)))
        return sqrt(x*x + y*y);
    return 0.0;
}


int main(int argc, char *argv[]){
    if (argc != 2){
        printf("Program receive %d numbers. Should be 1: epsilon\n", argc);
        return -1;
    }

    double epsilon = atof(argv[argc-1]);
    if (epsilon <= 0){
        printf("Epsilon should be > 0!!!\n");
        return -1;
    }

    int N;
    if (epsilon < 1e-7){
        printf("Set epsilon < 1e-7!");
        return -1;
    }
    N = roundf(1 / epsilon);

    int myrank, nprocess;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
    if (nprocess < 2){
        printf("Too few processes!");
        MPI_Finalize();
        return -1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int block_size = N / (nprocess-1);
    double (*p)[3] = malloc(sizeof(double[block_size][3]));
    //double *p = malloc(sizeof(double) * 3 * block_size);

    double global_integral;
    double current_sum = 0.;
    int converge = 0;
    double error;
    double total_points_amount = 0.;
    double whole_time = 0.;
    double start, finish, slave_time;
    start = MPI_Wtime();

    while (!converge) {
        if (!myrank) {
            total_points_amount += ((nprocess - 1) * block_size);
//            srand(time(NULL)); //Unnecessary string
            for (int i = 1; i < nprocess; i++) {
                for (int j = 0; j < block_size; j++) {
                    p[j][0] = 2 * (double) rand() / RAND_MAX - 1;
                    p[j][1] = 2 * (double) rand() / RAND_MAX - 1;
                    p[j][2] = (double) rand() / RAND_MAX;
                }

                MPI_Send(p, block_size * 3, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            }

            double total_sum = 0.0;
//        double local_sum;
//        MPI_Status status;
//        for (int i = 1; i < nprocess; total_sum += local_sum, ++i)
//            MPI_Recv(&local_sum, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);

            MPI_Reduce(MPI_IN_PLACE, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            MPI_Reduce(MPI_IN_PLACE, &whole_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            current_sum += total_sum;

            global_integral = current_sum / total_points_amount;
            global_integral *= 4.;

            error = fabs(global_integral - (M_PI / 6));
            // whole_time += max_time;
            if (error < epsilon)
                converge = 1;

            for (int i = 1; i < nprocess; i++) {
                MPI_Send(&converge, 1, MPI_INT, i, i, MPI_COMM_WORLD);
            }
        } else {
            MPI_Status status;
            MPI_Recv(p, block_size * 3, MPI_DOUBLE, 0, myrank, MPI_COMM_WORLD, &status);

            double sum = 0.0;
            for (int j = 0; j < block_size; j++)
                sum += func(p[j][0], p[j][1], p[j][2]);

            // MPI_Send(&sum, 1, MPI_DOUBLE, 0, myrank, MPI_COMM_WORLD);
            MPI_Reduce(&sum, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            finish = MPI_Wtime();
            slave_time = finish - start;

            MPI_Reduce(&slave_time, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Recv(&converge, 1, MPI_INT, 0, myrank, MPI_COMM_WORLD, &status);
        }
    }
    if (!myrank){
        printf("True integral value=%.10f\n", M_PI/6);
        printf("Total Monte Carlo estimate=%.10f\n", global_integral);
        printf("N points=%d\n", (int)total_points_amount);
        printf("Final absolute error=%.10f\n", error);
        printf("Total time=%.10f sec\n", whole_time);
    }
    free(p);
    MPI_Finalize();
    return 0;
}
