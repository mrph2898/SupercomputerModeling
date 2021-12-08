#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>
#include <math.h>
#include <unistd.h>


double u_2(double x, double y){
    return sqrt(4 + x * y);
}


double k_3(double x, double y){
    return 4 + x + y;
}


double q_2(double x, double y){
    double sum = x + y;
    if (sum < 0) {
        return 0;
    } else {
        return sum;
    }
}


double F(double x, double y){
    return ((pow(x, 3) - x*x*(y - 4) - x*(y*y + 8) +
    y*(y*y + 4*y - 8) + 4*q_2(x, y)*pow((4 + x*y), 2)) /
    (4 * pow((4 + x*y), 1.5)));
}


double psi_R(double x, double y){
    return (-y*y*(4 + x + y) + 4*pow(4 + x*y, 2)) / (4*pow(4 + x*y, 1.5));
}


double psi_L(double x, double y){
    return (y*y*(4 + x + y) + 4*pow(4 + x*y, 2)) / (4*pow(4 + x*y, 1.5));
}


double psi_T(double x, double y){
    return (-x*x*(4 + x + y)) / (4*pow(4 + x*y, 1.5));
}


double psi_B(double x, double y){
    return -psi_T(x, y);
}


double dot_product(int M, int N,
                   double (*U)[N + 2], double (*V)[N + 2],
                   double h1, double h2
                   ){
    double answer = 0.;
    for (int i=1; i <= M; i++){
        for (int j=1; j <= N; j++){
            double rho, rho1, rho2;

            if ((i == 1) || (i == M)){
                rho1 = 0.5;
            } else {
                rho1 = 1;
            }
            if ((j == 1) || (j == N)){
                rho2 = 0.5;
            } else {
                rho2 = 1;
            }
            rho = rho1 * rho2;
            answer += (rho * U[i][j] * V[i][j] * h1 * h2);
        }
    }
    return answer;
}


double norm(int M, int N, double (*U)[N + 2],
            double h1, double h2){
    return sqrt(dot_product(M, N, U, U, h1, h2));
}


void B_right(int M, int N, double (*B)[N+2],
       double h1, double h2,
       double x_start, double y_start,
       int left_border, int right_border,
       int top_border, int bottom_border){
    int i, j;
# pragma omp parallel for private(i, j)
    for(i = 1; i <= M; i++)
        for (j = 1; j <= N; j++)
            B[i][j] = F(x_start + i * h1, y_start + j * h2);

    if (left_border){
        for (j = 2; j < N; j++) {
            B[1][j] = (F(x_start + h1, y_start + j * h2) +
                    psi_L(x_start + h1, y_start + j * h2) * 2/h1);
        }
        if (top_border){
            B[1][N] = F(x_start + h1, y_start + N*h2) + (2/h1 +  2/h2) * u_2(x_start + h1, y_start + N*h2);
        }
        if (bottom_border){
            B[1][1] = F(x_start + h1, y_start + h2) + (2/h1 +  2/h2) * u_2(x_start + h1, y_start + h2);
        }
    }

    if (right_border){
        for (j = 2; j < N; j++) {
            B[M][j] = (F(x_start + M*h1, y_start + j * h2) +
                       psi_R(x_start + M*h1, y_start + j * h2) * 2/h1);
        }
        if (top_border){
            B[M][N] = F(x_start + M*h1, y_start + N*h2) + (2/h1 +  2/h2) * u_2(x_start + M*h1, y_start + N*h2);
        }
        if (bottom_border){
            B[M][1] = F(x_start + M*h1, y_start + h2) + (2/h1 +  2/h2) * u_2(x_start + M*h1, y_start + h2);
        }
    }

    if (top_border){
        for (i = 2; i < M; i++) {
            B[i][N] = (F(x_start + i*h1, y_start + N*h2) +
                       psi_T(x_start + i*h1, y_start + N*h2) * 2/h2);
        }
    }

    if (bottom_border){
        for (i = 2; i < M; i++) {
            B[i][1] = (F(x_start + i*h1, y_start + h2) +
                       psi_B(x_start + i*h1, y_start + h2) * 2/h2);
        }
    }
}


double aw_ij(int N,
             double (*w)[N+2],
             double x_start, double y_start,
             int i, int j,
             double h1, double h2
             ){
    return (1/h1) * (k_3(x_start + (i + 0.5) * h1, y_start + j * h2) * (w[i + 1][j] - w[i][j]) / h1
    - k_3(x_start + (i - 0.5) * h1, y_start + j * h2) * (w[i][j] - w[i - 1][j]) / h1);
}

double bw_ij(int N,
             double (*w)[N+2],
             double x_start, double y_start,
             int i, int j,
             double h1, double h2
){
    return (1/h2) * (k_3(x_start + i * h1, y_start + (j + 0.5) * h2) * (w[i][j + 1] - w[i][j]) / h2
    - k_3(x_start + i * h1, y_start + (j - 0.5) * h2) * (w[i][j] - w[i][j - 1]) / h2);
}

void Aw_mult(int M, int N,
             double (*A)[N+2], double (*w)[N+2],
             double h1, double h2,
             double x_start, double y_start,
             int left_border, int right_border,
             int top_border, int bottom_border
             )
{
    double aw_x, bw_y;
    int i, j;
    for (i = 1; i <= M; i++){
        for (j = 1; j <= N; j++) {
            aw_x = aw_ij(N, w, x_start, y_start, i, j, h1, h2);
            bw_y = bw_ij(N, w, x_start, y_start, i, j, h1, h2);
            A[i][j] = -aw_x - bw_y + q_2(x_start + i * h1, y_start + j * h2) * w[i][j];
        }
    }

    // Left interior border filling
    if (left_border){
        for (j = 2; j < N; j++) {
            aw_x = aw_ij(N, w, x_start, y_start, 2, j, h1, h2);
            bw_y = bw_ij(N, w, x_start, y_start, 1, j, h1, h2);
            A[1][j] = -2*aw_x / h1 - bw_y + (q_2(x_start + h1, y_start + j * h2) + 2/h1) * w[1][j];
        }
        if (bottom_border){
            aw_x = aw_ij(N, w, x_start, y_start, 2, 1, h1, h2);
            bw_y = bw_ij(N, w, x_start, y_start, 1, 2, h1, h2);
            A[1][1] = -2*aw_x / h1 - 2 * bw_y / h2 + (q_2(x_start + h1, y_start + h2) + 2/h1) * w[1][1];
        }
        if (top_border){
            aw_x = aw_ij(N, w, x_start, y_start, 2, N, h1, h2);
            bw_y = bw_ij(N, w, x_start, y_start, 1, N, h1, h2);
            A[1][N] = -2*aw_x / h1 + 2*bw_y / h2 + (q_2(x_start + h1, y_start + N * h2) + 2/h1)* w[1][N];
        }
    }


    // Right interior border
    if (right_border){
        for (j = 2; j < N; j++) {
            aw_x = aw_ij(N, w, x_start, y_start, M, j, h1, h2);
            bw_y = bw_ij(N, w, x_start, y_start, M, j, h1, h2);
            A[M][j] = 2*aw_x / h1 - bw_y + (q_2(x_start + M * h1, y_start + j * h2) + 2/h1) * w[M][j];
        }
        if (bottom_border){
            aw_x = aw_ij(N, w, x_start, y_start, M, 1, h1, h2);
            bw_y = bw_ij(N, w, x_start, y_start, M, 2, h1, h2);
            A[M][1] = 2*aw_x / h1 - 2 * bw_y / h2 + (q_2(x_start + M * h1, y_start + h2) + 2/h1) * w[M][1];
        }
        if (top_border) {
            aw_x = aw_ij(N, w, x_start, y_start, M, N, h1, h2);
            bw_y = bw_ij(N, w, x_start, y_start, M, N, h1, h2);
            A[M][N] = 2*aw_x / h1 + 2 * bw_y / h2 + (q_2(x_start + M * h1, y_start + N * h2) + 2/h1) * w[M][N];
        }
    }

    // Top border
    if (top_border){
        for (i = 2; i < M; i++) {
            aw_x = aw_ij(N, w, x_start, y_start, i, N, h1, h2);
            bw_y = bw_ij(N, w, x_start, y_start, i, N, h1, h2);
            A[i][N] = -aw_x + 2*bw_y / h2 + q_2(x_start + i * h1, y_start + N * h2) * w[i][N];
        }
    }

    // Bottom border
    if (bottom_border){
        for (i = 2; i < M; i++) {
            aw_x = aw_ij(N, w, x_start, y_start, i, 1, h1, h2);
            bw_y = bw_ij(N, w, x_start, y_start, i, 2, h1, h2);
            A[i][1] = -aw_x - 2*bw_y / h2 + q_2(x_start + i * h1, y_start + h1) * w[i][1];
        }
    }
}


void calculate_r(int M, int N,
                 double (*r)[N+2],
                 double (*Aw)[N+2],
                 double (*B)[N+2]
                 ){
    int i, j;
    //# pragma omp parallel for private(i, j)
    for(i = 0; i <= M + 1; i++)
    {
        for (j = 0; j <= N + 1; j++)
        {
            if(i <= 1 || i >= M || j <= 1 || j >= N)
                r[i][j] = 0;
            else
                r[i][j] = Aw[i][j] - B[i][j];
        }
    }
}


void get_idx_n_idx(int *idx,
                   int *n_idx,
                   int process_amnt,
                   int grid_size,
                   int coordinate){
    if (grid_size % process_amnt == 0)
    {
        *n_idx = grid_size / process_amnt;
        *idx = coordinate * (grid_size / process_amnt);
    }
    else
    {
        if (coordinate == 0)
        {
            *n_idx = grid_size % process_amnt + grid_size / process_amnt;
            *idx = 0;
        }
        else
        {
            *n_idx = grid_size / process_amnt;
            *idx = grid_size % process_amnt + coordinate * (grid_size / process_amnt);
        }
    }
}


#define A1 0.0
#define A2  4.0
#define B1  0.0
#define B2  3.0
#define EPS_REL  0.00001
#define DOWN_TAG 1000
#define MAX_ITER 3000


void print_matrix(int M, int N, double (*A)[N+2]){
    printf("Matrix:\n");
    for (int i=0; i<=M+1; i++){
        for (int j=0; j<=N+1; j++){
            printf("%3.2f ", A[i][j]);
        }
        printf("\n");
    }
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Program receive %d numbers. Should be 2: M, N\n", argc);
        return -1;
    }

    int M = atoi(argv[argc - 2]);
    int N = atoi(argv[argc - 1]);
    if ((M <= 0) || (N <= 0)) {
        printf("M and N should be integer and > 0!!!\n");
        return -1;
    }
    printf("M = %d, N = %d\n", M, N);

    int my_rank;
    int n_processes;
    int process_amounts[2] = {0, 0};
    int write[1] = {0};

    double h1 = (A2 - A1) / M;
    double h2 = (B2 - B1) / N;
    double cur_eps = 1.0;

    MPI_Init(&argc, &argv);
    MPI_Status status;
    MPI_Request request;

    // For the cartesian topology
    MPI_Comm MPI_COMM_CART;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    // Creating rectangular supports
    MPI_Dims_create(n_processes, 2, process_amounts);

    printf("p_x = %d, p_y = %d\n", process_amounts[0], process_amounts[1]);
    int periods[2] = {0, 0};

    // Create cartesian  topology in communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2,
                    process_amounts, periods,
                    1, &MPI_COMM_CART);

    int my_coords[2];
    // Receive corresponding to rank process coordinates
    MPI_Cart_coords(MPI_COMM_CART, my_rank, 2, my_coords);

    int x_idx, n_x;
    get_idx_n_idx(&x_idx, &n_x, process_amounts[0], M, my_coords[0]);

    int y_idx, n_y;
    get_idx_n_idx(&y_idx, &n_y, process_amounts[1], N, my_coords[1]);

    double start_time = MPI_Wtime();

    // Create each block of size n_x+2 and n_y+2 with borders
    double *t_send = malloc(sizeof(double[n_x]));
    double *t_rec = malloc(sizeof(double[n_x]));

    double *b_send = malloc(sizeof(double[n_x]));
    double *b_rec = malloc(sizeof(double[n_x]));

    double *l_send = malloc(sizeof(double[n_y]));
    double *l_rec = malloc(sizeof(double[n_y]));

    double *r_send = malloc(sizeof(double[n_y]));
    double *r_rec = malloc(sizeof(double[n_y]));

    int i, j;
    int n_iters = 0;
    double block_eps;

    double (*w)[n_y + 2] = malloc(sizeof(double[n_x + 2][n_y + 2]));
    double (*w_pr)[n_y + 2] = malloc(sizeof(double[n_x + 2][n_y + 2]));
    double (*B)[n_y + 2] = malloc(sizeof(double[n_x + 2][n_y + 2]));
    double tau;
    double eps_local, eps_r;

    double (*Aw)[n_y + 2] = malloc(sizeof(double[n_x + 2][n_y + 2]));
    double (*r)[n_y + 2] = malloc(sizeof(double[n_x + 2][n_y + 2]));
    double (*Ar)[n_y + 2] = malloc(sizeof(double[n_x + 2][n_y + 2]));
    double (*w_w_pr)[n_y + 2] = malloc(sizeof(double[n_x + 2][n_y + 2]));

    int left_border = 0;
    int top_border = 0;
    int right_border = 0;
    int bottom_border = 0;

    if (my_coords[0] == 0){
        left_border = 1;
    } else if (my_coords[0] == (process_amounts[0] - 1)){
        right_border = 1;
    }

    if (my_coords[1] == 0){
        bottom_border = 1;
    } else if (my_coords[1] == (process_amounts[1] - 1)){
        top_border = 1;
    }

    double (*Au)[n_y + 2] = malloc(sizeof(double[n_x + 2][n_y + 2]));
    double (*U)[n_y + 2] = malloc(sizeof(double[n_x + 2][n_y + 2]));
    for (i = 0; i <= n_x + 1; i++)
        for (j = 0; j <= n_y + 1; j++)
            U[i][j] = u_2(A1 + (x_idx + i) * h1, B1 + (y_idx + j) * h2);

    Aw_mult(n_x, n_y, Au, U, h1, h2, A1 + x_idx * h1, B1 + y_idx * h2,
            left_border, right_border,
            top_border,  bottom_border);
# pragma omp parallel for private(i, j)
    for (i = 0; i <= n_x + 1; i++)
        for (j = 0; j <= n_y + 1; j++)
            w[i][j] = 0;

    B_right(n_x, n_y, B,
            h1, h2,
            A1 + x_idx * h1,
            B1 + y_idx * h2,
            left_border, right_border,
            top_border,  bottom_border);
    double error_mean = 0;
    for (i = 1; i <= n_x; i++)
        for (j = 1; j <= n_y; j++){
            error_mean += fabs(Au[i][j] - B[i][j]);
        }
    printf("ERROR FROM B = %3.2f\n", error_mean / ((n_x) * (n_y)));

    while ((cur_eps > EPS_REL) && (n_iters < MAX_ITER)) {
//        print_matrix(M, N, w);
        n_iters++;
        // # pragma omp parallel for private(i, j)
        for (i = 0; i <= n_x + 1; i++) {
            for (j = 0; j <= n_y + 1; j++) {
                if (i == 0 || j == 0 || i == n_x + 1 || i == n_y + 1) {
                    w_pr[i][j] = 0;
                } else {
                    w_pr[i][j] = w[i][j];
                }
            }
        }

        int neighbour_rank;
        int neighbour_coords[2];
        int tag = 0;

        // Bottom border
        if ((process_amounts[1] > 1) && (my_coords[1] != 0)) {
            for (int i = 0; i < n_x; i++)
                b_send[i] = w[i+1][0];

            neighbour_coords[0] = my_coords[0];
            neighbour_coords[1] = my_coords[1] - 1;

            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Isend(b_send, n_x, MPI_DOUBLE,
                      neighbour_rank, tag + DOWN_TAG,
                      MPI_COMM_CART, &request);
        }

        // Left border send
        if ((process_amounts[0] > 1) && (my_coords[0] != 0)) {
            for (int j = 0; j < n_y; j++)
                l_send[j] = w[n_x][j+1];

            neighbour_coords[0] = my_coords[0] - 1;
            neighbour_coords[1] = my_coords[1];

            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Isend(l_send, n_y, MPI_DOUBLE,
                      neighbour_rank, tag,
                      MPI_COMM_CART, &request);
        }

        // Top border
        if ((process_amounts[1] > 1) && (my_coords[1] != (process_amounts[1] - 1))) {
            for (int i = 0; i < n_x; i++)
                t_send[i] = w[i+1][n_y];

            neighbour_coords[0] = my_coords[0];
            neighbour_coords[1] = my_coords[1] + 1;

            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Isend(t_send, n_x, MPI_DOUBLE,
                      neighbour_rank, tag,
                      MPI_COMM_CART, &request);
        }

        // Right border
        if ((process_amounts[0] > 1) && (my_coords[0] != (process_amounts[0] - 1))) {
            for (int j = 0; j < n_y; j++)
                r_send[j] = w[1][j+1];

            neighbour_coords[0] = my_coords[0] + 1;
            neighbour_coords[1] = my_coords[1];

            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Isend(r_send, n_y, MPI_DOUBLE,
                      neighbour_rank, tag,
                      MPI_COMM_CART, &request);
        }

        // Receive borders
        // Bottom border
        if ((bottom_border && (process_amounts[1] > 1)) || (process_amounts[1] == 1)) {
            for (int i = 1; i <= n_x; i++)
                w[i][0] = psi_B(A1 + (x_idx + i) * h1, B1);
        } else {
            neighbour_coords[0] = my_coords[0];
            neighbour_coords[1] = my_coords[1] - 1;
            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Recv(b_rec, n_x, MPI_DOUBLE,
                     neighbour_rank, tag, MPI_COMM_CART, &status);


            for (int i = 1; i <= n_x; i++)
                w[i][0] = b_rec[i];
        }

        // Left border
        if ((left_border && (process_amounts[0] > 1)) || (process_amounts[0] == 1)) {
            for (int j = 1; j <= n_y; j++)
                w[0][j] = psi_L(A1, B1 + (y_idx + j) * h2);
        } else {
            neighbour_coords[0] = my_coords[0] - 1;
            neighbour_coords[1] = my_coords[1];

            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Recv(l_rec, n_y, MPI_DOUBLE,
                     neighbour_rank, tag, MPI_COMM_CART, &status);

            for (int j = 1; j <= n_y; j++)
                w[0][j] = l_rec[j];
        }

        // Top border
        if ((top_border && (process_amounts[1] > 1)) || (process_amounts[1] == 1)) {
            for (int i = 1; i <= n_x; i++)
                w[i][n_y + 1] = psi_T(A1 + (x_idx + i) * h1, B2);
        } else {
            neighbour_coords[0] = my_coords[0];
            neighbour_coords[1] = my_coords[1] + 1;
            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Recv(t_rec, n_x, MPI_DOUBLE,
                     neighbour_rank, tag + DOWN_TAG,
                     MPI_COMM_CART, &status);

            for (int i = 1; i <= n_x; i++)
                w[i][n_y + 1] = t_rec[i];
        }

        // Right border
        if ((right_border && (process_amounts[0] > 1)) || (process_amounts[0] == 1)) {
            for (int j = 1; j <= n_y; j++)
                w[n_x + 1][j] = psi_R(A2, B1 + (y_idx + j) * h2);
        } else {
            neighbour_coords[0] = my_coords[0] + 1;
            neighbour_coords[1] = my_coords[1];
            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Recv(r_rec, n_y, MPI_DOUBLE,
                     neighbour_rank, tag, MPI_COMM_CART, &status);

            for (int j = 1; j <= n_y; j++)
                w[n_x + 1][j] = r_rec[j];
        }

        Aw_mult(n_x, n_y,
                Aw, w,
                h1, h2,
                A1 + x_idx * h1, B1 + y_idx * h2,
                left_border, right_border,
                top_border,  bottom_border);
        calculate_r(n_x, n_y, r, Aw, B);
        Aw_mult(n_x, n_y,
                Ar, r,
                h1, h2,
                A1 + x_idx * h1, B1 + y_idx * h2,
                left_border, right_border,
                top_border,  bottom_border);

        tau = dot_product(n_x, n_y, Ar, r, h1, h2) / norm(n_x, n_y, Ar, h1, h2);
        printf("Tau = %f\n", tau);
# pragma omp parallel for private(i, j)
        for (i = 1; i <= n_x; i++)
            for (j = 1; j <= n_y; j++)
                w[i][j] = w[i][j] - tau * r[i][j] / 2;

        calculate_r(n_x, n_y, w_w_pr, w, w_pr);
        block_eps = norm(n_x, n_y, w_w_pr, h1, h2);

        MPI_Allreduce(&block_eps, &cur_eps, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // Waiting for all processes
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (my_rank != 0) {
        MPI_Recv(write, 1, MPI_INT, my_rank - 1, 0, MPI_COMM_WORLD, &status);
    } else {
        printf("TIME = %f\n", end_time - start_time);
        printf("Number of iterations = %d\n", n_iters);
        printf("Tau = %f\n", tau);
        printf("Eps = %f\n", EPS_REL);
    }

    usleep(500);
    if (my_rank != n_processes - 1)
        MPI_Send(write, 1, MPI_INT, my_rank + 1, 0, MPI_COMM_WORLD);

    free(Au);
    free(U);
    free(w);
    free(w_pr);
    free(B);
    free(r);
    free(Ar);
    free(Aw);
    free(w_w_pr);

    free(t_send);
    free(t_rec);
    free(b_send);
    free(b_rec);
    free(r_send);
    free(r_rec);
    free(l_send);
    free(l_rec);
    MPI_Finalize();
    return 0;
}