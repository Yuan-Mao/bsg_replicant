// Copyright (c) 2019, University of Washington All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this list
// of conditions and the following disclaimer.
//
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
//
// Neither the name of the copyright holder nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <bsg_manycore_regression.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>

using namespace std;

#define PRINT_SCORE
//#define PRINT_MATRIX
#define ALLOC_NAME "default_allocator"
#define CUDA_CALL(expr)                                                 \
        {                                                               \
                int __err;                                              \
                __err = expr;                                           \
                if (__err != HB_MC_SUCCESS) {                           \
                        bsg_pr_err("'%s' failed: %s\n", #expr, hb_mc_strerror(__err)); \
                        return __err;                                   \
                }                                                       \
        }

int kernel_smith_waterman (int argc, char **argv) {
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the CUDA Vector Addition Kernel on one 2x2 tile groups.\n");

        srand(static_cast<unsigned>(time(0)));

        /* Define path to binary. */
        /* Initialize device, load binary and unfreeze tiles. */
        hb_mc_dimension_t tg_dim = { .x = 16, .y = 8};
        hb_mc_device_t device;
        BSG_CUDA_CALL(hb_mc_device_init_custom_dimensions(&device, test_name, 0, tg_dim));

        /* if DMA is not supported just return SUCCESS */
        if (!hb_mc_manycore_supports_dma_write(device.mc)
            || !hb_mc_manycore_supports_dma_read(device.mc)) {
                bsg_pr_test_info("DMA not supported for this machine: returning success\n");
                BSG_CUDA_CALL(hb_mc_device_finish(&device));
                return HB_MC_SUCCESS;
        }

        hb_mc_pod_id_t pod;
        hb_mc_device_foreach_pod_id(&device, pod)
        {
                BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));
                BSG_CUDA_CALL(hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0));

                // == Reading data ==
                const int match_score    = 1;
                const int mismatch_score = -3;
                const int gap_open       = -3;
                const int gap_extend     = -1;
                ifstream f_ref, f_query;
                ofstream fout;

                f_ref.open("data/dna-reference.fasta", ios::in);
                f_query.open("data/dna-query.fasta", ios::in);
                fout.open("output", ios::out);

                const int N = 128;
                string str, num;
                string seqa_str = "";
                string seqb_str = "";
                int* sizea = new int[N];
                int* sizeb = new int[N];

                // read N sequences from file
                for (int i = 0; i < N; i++) {
                  f_ref >> num >> str;
                  seqa_str += str;
                  sizea[i] = str.size();
                  f_query >> num >> str;
                  sizeb[i] = str.size();
                  seqb_str += str;
                };

                const char* seqa = seqa_str.c_str();
                const char* seqb = seqb_str.c_str();

                f_ref.close();
                f_query.close();
                fout.close();

                // == Sending data to device

                // Define the sizes of the I/O arrays
                size_t seqa_bytes = seqa_str.size() * sizeof(char);
                size_t seqb_bytes = seqb_str.size() * sizeof(char);
                size_t sizea_bytes = N * sizeof(int);
                size_t sizeb_bytes = N * sizeof(int);
                size_t score_bytes = N * sizeof(int);

                // Allocate device memory for the I/O arrays
                eva_t seqa_d, seqb_d, sizea_d, sizeb_d, score_d;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, seqa_bytes, &seqa_d));
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, seqb_bytes, &seqb_d));
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, sizea_bytes, &sizea_d));
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, sizeb_bytes, &sizeb_d));
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, score_bytes, &score_d));

                //BSG_CUDA_CALL(hb_mc_device_malloc(&device, vsize3, &n1_device));

                //BSG_CUDA_CALL(hb_mc_device_malloc(&device, vsize3, &n2_device));

                /* Copy A & B from host onto device DRAM. */
                //hb_mc_dma_htod_t htod_jobs [] = {
                        //{
                                //.d_addr = ref_device,
                                //.h_addr = ref,
                                //.size   = vsize0
                        //},
                        //{
                                //.d_addr = query_device,
                                //.h_addr = query,
                                //.size   = vsize1
                        //},
                        //{
                                //.d_addr = n1_device,
                                //.h_addr = n1,
                                //.size   = vsize3
                        //},
                        //{
                                //.d_addr = n2_device,
                                //.h_addr = n2,
                                //.size   =  vsize3
                        //}
                //};

                //bsg_pr_test_info("Writing A and B to device\n");

                //BSG_CUDA_CALL(hb_mc_device_dma_to_device(&device, htod_jobs, 4));

                /* Define block_size_x/y: amount of work for each tile group */
                /* Define tg_dim_x/y: number of tiles in each tile group */
                /* Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y */
                //hb_mc_dimension_t grid_dim = { .x = 1, .y = 1};


                /* Prepare list of input arguments for kernel. */
                // N1 is the number of alignments per tile while N is the numeber of total alignments
                //int N1 = 2;
                //uint32_t cuda_argv[6] = {ref_device, query_device, score_matrix_device, n1_device, n2_device, N1};

                /* Enqqueue grid of tile groups, pass in grid and tile group dimensions,
                   kernel name, number and list of input arguments */
                //BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_smith_waterman", 6, cuda_argv));

                /* Launch and execute all tile groups on device and wait for all to finish.  */
                //BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));

                /* Copy result matrix back from device DRAM into host memory.  */
                //int* score_matrix_host = new int[N*sm_size]();

                //hb_mc_dma_dtoh_t dtoh_job = {
                        //.d_addr = score_matrix_device,
                        //.h_addr = score_matrix_host,
                        //.size   = vsize2
                //};

                //bsg_pr_test_info("Reading C to host\n");

                //BSG_CUDA_CALL(hb_mc_device_dma_to_host(&device, &dtoh_job, 1));

                /* Calculate the expected result using host code and compare the results.  */
                //if (mismatch)
                        //return HB_MC_FAIL;

                /* Freeze the tiles and memory manager cleanup.  */
                //BSG_CUDA_CALL(hb_mc_device_program_finish(&device));

        }

        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_smith_waterman", kernel_smith_waterman);

