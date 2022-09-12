#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <string.h>

#include "cuda_profiler_api.h"
#include "blocks_functions.h"


void get_idx_n_idx(int *idx,
                   int *n_idx,
                   int process_amnt,
                   int grid_size,
                   int coordinate){
    if (grid_size % process_amnt == 0) {
        *n_idx = grid_size / process_amnt;
        *idx = coordinate * (grid_size / process_amnt);
    }
    else
    {
        if (coordinate == 0){
            *n_idx = grid_size % process_amnt + grid_size / process_amnt;
            *idx = 0;
        } else
        {
            *n_idx = grid_size / process_amnt;
            *idx = grid_size % process_amnt + coordinate * (grid_size / process_amnt);
        }
    }
}


void send_recv_borders(int n_x, int n_y,
                       const int process_amounts[2],
                       double x_idx,
                       double y_idx,
                       const int my_coords[2],
                       int tag,
                       double **w,
                       double *b_send,
                       double *l_send,
                       double *t_send,
                       double *r_send,
                       double *b_rec,
                       double *l_rec,
                       double *t_rec,
                       double *r_rec,
                       int left_border, int right_border,
                       int top_border, int bottom_border,
                       double h1, double h2,
                       MPI_Comm MPI_COMM_CART
){
    int neighbour_coords[2];
    int neighbour_rank;
//    int num_threads_x = (int) sqrt(threadsPerBlock);
//    int num_threads_y = threadsPerBlock / numThreadsX;
    int blocksPerGrid_x = n_x / numThreadsX + 1;
    int blocksPerGrid_y = n_y / numThreadsY + 1;
    dim3 gridShape = dim3(blocksPerGrid_x, blocksPerGrid_y);
    dim3 blockShape = dim3(numThreadsX, numThreadsY);

    MPI_Request request[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                              MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Status status;

    double *dev_b_send, *dev_l_send, *dev_t_send, *dev_r_send;
    double *dev_b_rec, *dev_l_rec, *dev_t_rec, *dev_r_rec;
    cudaMalloc((void**)&dev_b_send, sizeof(double[n_x]));
    cudaMalloc((void**)&dev_t_send, sizeof(double[n_x]));
    cudaMalloc((void**)&dev_b_rec, sizeof(double[n_x]));
    cudaMalloc((void**)&dev_t_rec, sizeof(double[n_x]));

    cudaMalloc((void**)&dev_l_send, sizeof(double[n_y]));
    cudaMalloc((void**)&dev_r_send, sizeof(double[n_y]));
    cudaMalloc((void**)&dev_l_rec, sizeof(double[n_y]));
    cudaMalloc((void**)&dev_r_rec, sizeof(double[n_y]));
    /////////////////
    // Bottom border send
    if ((process_amounts[1] > 1) && !bottom_border) {
        get_bottom<<<gridShape, blockShape>>>(n_x, n_y,
                                           x_idx, y_idx,
                                           w, dev_b_send);
        cudaMemcpy(b_send, dev_b_send,
                   sizeof(double[n_x]), cudaMemcpyDeviceToHost);

        neighbour_coords[0] = my_coords[0];
        neighbour_coords[1] = my_coords[1] - 1;

        MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
        MPI_Isend(b_send, n_x, MPI_DOUBLE,
                  neighbour_rank, tag + DOWN_TAG,
                  MPI_COMM_CART, &request[0]);
    }

    // Left border send
    if ((process_amounts[0] > 1) && !left_border) {
        get_left<<<gridShape, blockShape>>>(n_x, n_y,
                                              x_idx, y_idx,
                                              w, dev_l_send);
        cudaMemcpy(l_send, dev_l_send,
                   sizeof(double[n_y]), cudaMemcpyDeviceToHost);

        neighbour_coords[0] = my_coords[0] - 1;
        neighbour_coords[1] = my_coords[1];

        MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
        MPI_Isend(l_send, n_y, MPI_DOUBLE,
                  neighbour_rank, tag,
                  MPI_COMM_CART, &request[1]);
    }

    // Top border
    if ((process_amounts[1] > 1) && !top_border) {
        get_top<<<gridShape, blockShape>>>(n_x, n_y,
                                            x_idx, y_idx,
                                            w, dev_t_send);
        cudaMemcpy(t_send, dev_t_send,
                   sizeof(double[n_x]), cudaMemcpyDeviceToHost);

        neighbour_coords[0] = my_coords[0];
        neighbour_coords[1] = my_coords[1] + 1;

        MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
        MPI_Isend(t_send, n_x, MPI_DOUBLE,
                  neighbour_rank, tag,
                  MPI_COMM_CART, &request[2]);
    }

    // Right border
    if ((process_amounts[0] > 1) && !right_border) {
        get_right<<<gridShape, blockShape>>>(n_x, n_y,
                                            x_idx, y_idx,
                                            w, dev_r_send);
        cudaMemcpy(r_send, dev_r_send,
                   sizeof(double[n_y]), cudaMemcpyDeviceToHost);

        neighbour_coords[0] = my_coords[0] + 1;
        neighbour_coords[1] = my_coords[1];

        MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
        MPI_Isend(r_send, n_y, MPI_DOUBLE,
                  neighbour_rank, tag,
                  MPI_COMM_CART, &request[3]);
    }

    // Receive borders
    // Bottom border
    if ((bottom_border && (process_amounts[1] > 1)) || (process_amounts[1] == 1)) {
        preset_bottom<<<gridShape, blockShape>>>(n_x, n_y, x_idx, y_idx, w, h1, h2);
    } else {
        neighbour_coords[0] = my_coords[0];
        neighbour_coords[1] = my_coords[1] - 1;
        MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
        MPI_Recv(b_rec, n_x, MPI_DOUBLE,
                 neighbour_rank, tag, MPI_COMM_CART, &status);

        cudaMemcpy(dev_b_rec, b_rec,
                   sizeof(double[n_x]), cudaMemcpyHostToDevice);
        set_bottom<<<gridShape, blockShape>>>(n_x, n_y, x_idx, y_idx, w, dev_b_rec);
    }

    // Left border
    if ((left_border && (process_amounts[0] > 1)) || (process_amounts[0] == 1)) {
        preset_left<<<gridShape, blockShape>>>(n_x, n_y, x_idx, y_idx, w, h1, h2);

    } else {
        neighbour_coords[0] = my_coords[0] - 1;
        neighbour_coords[1] = my_coords[1];

        MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
        MPI_Recv(l_rec, n_y, MPI_DOUBLE,
                 neighbour_rank, tag, MPI_COMM_CART, &status);

        cudaMemcpy(dev_l_rec, l_rec,
                   sizeof(double[n_y]), cudaMemcpyHostToDevice);
        set_left<<<gridShape, blockShape>>>(n_x, n_y, x_idx, y_idx, w, dev_l_rec);
    }

    // Top border
    if ((top_border && (process_amounts[1] > 1)) || (process_amounts[1] == 1)) {
        preset_top<<<gridShape, blockShape>>>(n_x, n_y, x_idx, y_idx, w, h1, h2);
    } else {
        neighbour_coords[0] = my_coords[0];
        neighbour_coords[1] = my_coords[1] + 1;
        MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
        MPI_Recv(t_rec, n_x, MPI_DOUBLE,
                 neighbour_rank, tag + DOWN_TAG,
                 MPI_COMM_CART, &status);

        cudaMemcpy(dev_t_rec, t_rec,
                   sizeof(double[n_x]), cudaMemcpyHostToDevice);
        set_top<<<gridShape, blockShape>>>(n_x, n_y, x_idx, y_idx, w, dev_t_rec);
    }

    // Right border
    if ((right_border && (process_amounts[0] > 1)) || (process_amounts[0] == 1)) {
        preset_right<<<gridShape, blockShape>>>(n_x, n_y, x_idx, y_idx, w, h1, h2);
    } else {
        neighbour_coords[0] = my_coords[0] + 1;
        neighbour_coords[1] = my_coords[1];
        MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
        MPI_Recv(r_rec, n_y, MPI_DOUBLE,
                 neighbour_rank, tag, MPI_COMM_CART, &status);

        cudaMemcpy(dev_r_rec, r_rec,
                   sizeof(double[n_y]), cudaMemcpyHostToDevice);
        set_right<<<gridShape, blockShape>>>(n_x, n_y, x_idx, y_idx, w, dev_r_rec);
    }

    for (int i = 0; i < 4; i++) {
        MPI_Wait(&request[i], &status);
    }
}


void cudaDotProduct(int n_x, int n_y,
                    int x_idx, int y_idx,
                    double **U, double **V,
                    double h1, double h2,
                    int left_border, int right_border,
                    int top_border, int bottom_border,
                    double *curr_sum)
{
//    int num_threads_x = (int) sqrt(threadsPerBlock);
//    int num_threads_y = threadsPerBlock / numThreadsX;
    int blocksPerGrid_x = n_x / numThreadsX + 1;
    int blocksPerGrid_y = n_y / numThreadsY + 1;
    dim3 gridShape = dim3(blocksPerGrid_x, blocksPerGrid_y);
    dim3 blockShape = dim3(numThreadsX, numThreadsY);
    /////////
    double c, *partial_c;
    double *dev_partial_c;
    partial_c = (double*) calloc(blocksPerGrid_x, sizeof(double));
    // Allocate device memory
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid_x * sizeof(double));
    /////////
    cuda_dot_product<<<gridShape, blockShape>>>(n_x, n_y, x_idx, y_idx,
                                                U, V, h1, h2,
                                                left_border, right_border,
                                                top_border, bottom_border,
                                                dev_partial_c);
    cudaMemcpy(partial_c, dev_partial_c,
               blocksPerGrid_x * sizeof(double),
               cudaMemcpyDeviceToHost);
    c = 0;
    for (int i = 0; i < blocksPerGrid_x; ++i) {
        c = c + partial_c[i];
    }
    (*curr_sum) = c;
    /////////
    cudaFree(dev_partial_c);
    free(partial_c);
    return;
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
    int my_rank, n_processes;
    int process_amounts[2] = {0, 0};
    int write[1] = {0};
    double h1 = (A2 - A1) / M;
    double h2 = (B2 - B1) / N;
    double cur_eps = 1.0;

    MPI_Init(&argc, &argv);
    MPI_Status status;

    // For the cartesian topology
    MPI_Comm MPI_COMM_CART;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    // Creating rectangular supports
    MPI_Dims_create(n_processes, 2, process_amounts);
    int periods[2] = {0, 0};

    // Create cartesian  topology in communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2,
                    process_amounts, periods,
                    1, &MPI_COMM_CART);

    int my_coords[2];
    // Receive corresponding to rank process coordinates
    MPI_Cart_coords(MPI_COMM_CART, my_rank, 2, my_coords);

    int x_idx, n_x;
    get_idx_n_idx(&x_idx, &n_x, process_amounts[0], M+1, my_coords[0]);

    int y_idx, n_y;
    get_idx_n_idx(&y_idx, &n_y, process_amounts[1], N+1, my_coords[1]);

    double start_time = MPI_Wtime();
    ////////////////////////
    cudaProfilerStart();
//    int num_threads_x = (int) sqrt(threadsPerBlock);
//    int num_threads_y = threadsPerBlock / numThreadsX;
    int blocksPerGrid_x = n_x / numThreadsX + 1;
    int blocksPerGrid_y = n_y / numThreadsY + 1;
    dim3 gridShape = dim3(blocksPerGrid_x, blocksPerGrid_y);
    dim3 blockShape = dim3(numThreadsX, numThreadsY);
    ////////////////////////
    double *t_send = (double *) malloc(sizeof(double[n_x]));
    double *t_rec = (double *) malloc(sizeof(double[n_x]));
    double *b_send = (double *) malloc(sizeof(double[n_x]));
    double *b_rec = (double *) malloc(sizeof(double[n_x]));

    double *l_send = (double *) malloc(sizeof(double[n_y]));
    double *l_rec = (double *) malloc(sizeof(double[n_y]));
    double *r_send = (double *) malloc(sizeof(double[n_y]));
    double *r_rec = (double *) malloc(sizeof(double[n_y]));
    int n_iters = 0;
    double block_eps;

    double **w, **w_pr, **B;
    double **Aw, **r_k, **Ar, **w_w_pr;

    cudaMalloc((void**)&w, sizeof(double[n_x + 2][n_y + 2]));
    cudaMalloc((void**)&w_pr, sizeof(double[n_x + 2][n_y + 2]));
    cudaMalloc((void**)&B, sizeof(double[n_x + 2][n_y + 2]));
    cudaMalloc((void**)&Aw, sizeof(double[n_x + 2][n_y + 2]));
    cudaMalloc((void**)&r_k, sizeof(double[n_x + 2][n_y + 2]));
    cudaMalloc((void**)&Ar, sizeof(double[n_x + 2][n_y + 2]));
    cudaMalloc((void**)&w_w_pr, sizeof(double[n_x + 2][n_y + 2]));
    ////////////////////////
    double tau = 0;
    double global_tau = 0;
    double denumenator;
    double whole_denum;
//    double global_alpha, global_beta;
//    double eps_local, eps_r;
    int left_border = 0;
    int top_border = 0;
    int right_border = 0;
    int bottom_border = 0;
    if (my_coords[0] == 0)
        left_border = 1;

    if (my_coords[0] == (process_amounts[0] - 1))
        right_border = 1;

    if (my_coords[1] == 0)
        bottom_border = 1;

    if (my_coords[1] == (process_amounts[1] - 1))
        top_border = 1;
    ////////////////////////
    cudaB_right<<<gridShape, blockShape>>>(n_x, n_y, B,
                                           x_idx, y_idx,
                                           h1, h2,
                                           A1 + x_idx * h1,
                                           B1 + y_idx * h2,
                                           left_border, right_border,
                                           top_border,  bottom_border);
    init_w<<<gridShape, blockShape>>>(n_x, w);

    int tag = 0;
    while ((cur_eps > EPS_REL) && (n_iters < MAX_ITER)) {
        if (my_rank == 0) {
            if (n_iters % 1000 == 0)
                printf("%g \n", cur_eps);
        }
        n_iters++;

        copy_interior_w<<<gridShape, blockShape>>>(n_x, n_y,
                                                   w, w_pr);

        send_recv_borders(n_x, n_y, process_amounts,
                          x_idx, y_idx, my_coords, tag,
                          w,
                          b_send, l_send, t_send, r_send,
                          b_rec, l_rec, t_rec, r_rec,
                          left_border, right_border,
                          top_border, bottom_border,
                          h1, h2, MPI_COMM_CART);
        cuda_Aw_mult<<<gridShape, blockShape>>>(n_x, n_y,
                    x_idx, y_idx,
                    Aw, w,
                    h1, h2,
                    A1 + x_idx * h1, B1 + y_idx * h2,
                    left_border, right_border,
                    top_border,  bottom_border);

        calculate_r<<<gridShape, blockShape>>>(n_x, n_y,
                                               x_idx, y_idx,
                                               r_k, Aw, B);
        send_recv_borders(n_x, n_y, process_amounts,
                          x_idx, y_idx, my_coords, tag,
                          r_k,
                          b_send, l_send, t_send, r_send,
                          b_rec, l_rec, t_rec, r_rec,
                          left_border, right_border,
                          top_border, bottom_border,
                          h1, h2, MPI_COMM_CART);
        cuda_Aw_mult<<<gridShape, blockShape>>>(n_x, n_y,
                     x_idx, y_idx,
                     Ar, r_k,
                     h1, h2,
                     A1 + x_idx * h1, B1 + y_idx * h2,
                     left_border, right_border,
                     top_border,  bottom_border);
        cudaDotProduct(n_x, n_y,
                     x_idx, y_idx,
                     Ar, r_k, h1, h2,
                     left_border, right_border,
                     top_border, bottom_border,
                     &tau);

        cudaDotProduct(n_x, n_y,
                       x_idx, y_idx,
                       Ar, Ar, h1, h2,
                       left_border, right_border,
                       top_border, bottom_border,
                       &denumenator);
        MPI_Allreduce(&tau,  &global_tau, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_CART);
        MPI_Allreduce(&denumenator,  &whole_denum, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_CART);
        global_tau = global_tau / whole_denum;
        cuda_w_step<<<gridShape, blockShape>>>(n_y,
                    x_idx, y_idx,
                    w, r_k,
                    tau
//                    w_next
                    );
        calculate_r<<<gridShape, blockShape>>>(n_x, n_y,
                                               x_idx, y_idx,
                                               w_w_pr, w, w_pr);

        cudaDotProduct(n_x, n_y,
                       x_idx, y_idx,
                       w_w_pr, w_w_pr, h1, h2,
                       left_border, right_border,
                       top_border, bottom_border,
                       &block_eps);
        block_eps = sqrt(block_eps);

        MPI_Allreduce(&block_eps, &cur_eps, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_CART);
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

    if (my_rank != n_processes - 1)
        MPI_Send(write, 1, MPI_INT, my_rank + 1, 0, MPI_COMM_WORLD);


    cudaFree(w);
    cudaFree(w_pr);
    cudaFree(B);
    cudaFree(Aw);
    cudaFree(r_k);
    cudaFree(Ar);
    cudaFree(w_w_pr);

    free(t_send);
    free(t_rec);
    free(b_send);
    free(b_rec);
    free(r_send);
    free(r_rec);
    free(l_send);
    free(l_rec);
    cudaProfilerStop();
    MPI_Finalize();
    return 0;
}