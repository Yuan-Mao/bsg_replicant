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
#include <bsg_manycore_responder.h>
#include <algorithm>
#include <vector>
#include <bsg_manycore_spsc_queue.hpp>

#define ALLOC_NAME "default_allocator"
#define TEST_BYTE 0xcd

#define BUFFER_ELS  10
#define CHAIN_LEN    4
#define NUM_PACKETS 100

/*!
 * Runs a host_stream kernel on a 2x2 tile group. 
 * Device allcoates memory on device and uses hb_mc_device_memset to set to a prefixed valu.
 * Device then calls an empty kernel and loads back the meomry to compare.
 * This tests uses the software/spmd/bsg_cuda_lite_runtime/host_stream/ Manycore binary in the BSG Manycore bitbucket repository.  
*/

//////////////////////////////////////////////////////
// Responder to check for packets from the manycore //
//////////////////////////////////////////////////////
static
hb_mc_request_packet_id_t resp_ids [] = {
    RQST_ID(RQST_ID_ANY_X, RQST_ID_ANY_Y, RQST_ID_ADDR(0x8888)),
    {/*sentinal*/},
};

static int resp_init(hb_mc_responder_t *resp, hb_mc_manycore_t *mc)
{
    return HB_MC_SUCCESS;
}

static int resp_quit(hb_mc_responder_t *resp, hb_mc_manycore_t *mc)
{
    return HB_MC_SUCCESS;
}

static std::vector<int> pkt_data;
static int resp_respond(hb_mc_responder_t *resp, hb_mc_manycore_t *mc, const hb_mc_request_packet_t *rqst)
{
    bsg_pr_info("%s: received packet %d from (%3d,%3d)\n"
               , __func__
               , static_cast<int>(hb_mc_request_packet_get_data(rqst))
               , rqst->x_src
               , rqst->y_src);

    pkt_data.push_back(static_cast<int>(hb_mc_request_packet_get_data(rqst)));
    return HB_MC_SUCCESS;
}


static
hb_mc_responder_t resp ("host-stream-test", resp_ids, resp_init, resp_quit, resp_respond);
source_responder(resp);


int kernel_host_stream(int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the CUDA Device Memset Kernel on a grid of one 2x2 tile group.\n\n");

        /*****************************************************************************************************************
        * Define path to binary.
        * Initialize device, load binary and unfreeze tiles.
        ******************************************************************************************************************/
        hb_mc_device_t *device = (hb_mc_device_t *) malloc(sizeof(hb_mc_device_t));
        BSG_CUDA_CALL(hb_mc_device_init(device, test_name, 0));
        BSG_CUDA_CALL(hb_mc_device_program_init(device, bin_path, ALLOC_NAME, 0));
        hb_mc_manycore_t *mc = device->mc;
        hb_mc_pod_id_t pod_id = device->default_pod_id;
        hb_mc_pod_t *pod = &device->pods[pod_id];

        /*****************************************************************************************************************
        * 
        ******************************************************************************************************************/
        eva_t buffer_device;
        eva_t count_device;
        BSG_CUDA_CALL(hb_mc_device_malloc(device, BUFFER_ELS * (CHAIN_LEN+1) * sizeof(int), &buffer_device));
        BSG_CUDA_CALL(hb_mc_device_malloc(device, (CHAIN_LEN+1) * sizeof(int), &count_device));

        BSG_CUDA_CALL(hb_mc_device_memset(device, &count_device, 0, (CHAIN_LEN+1) * sizeof(int)));

        int buffer_host [NUM_PACKETS];
        for (int i = 0; i < NUM_PACKETS; i++)
        {
            buffer_host[i] = i;
        }

        /*****************************************************************************************************************
        * Define block_size_x/y: amount of work for each tile group
        * Define tg_dim_x/y: number of tiles in each tile group
        * Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y
        ******************************************************************************************************************/
        hb_mc_dimension_t tg_dim = { .x = CHAIN_LEN, .y = 1 }; 

        hb_mc_dimension_t grid_dim = { .x = 1, .y = 1 };


        /*****************************************************************************************************************
        * Prepare list of input arguments for kernel.
        ******************************************************************************************************************/
        uint32_t cuda_argv[2] = {buffer_device, count_device};

        /*****************************************************************************************************************
        * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
        ******************************************************************************************************************/
        BSG_CUDA_CALL(hb_mc_kernel_enqueue (device, grid_dim, tg_dim, "kernel_host_stream", 2, cuda_argv));

        /*****************************************************************************************************************
        * Launch and execute all tile groups on device and wait for all to finish. 
        ******************************************************************************************************************/
        //  Constantly polls
        //  Instead call user callback while polling
        //  Context switching or co-routines
        //  TBB 
        //  Interrupt when packet gets received, can do bare metal
        //  Thread job to periodically read off the queue
        //  Callback hook in inner loop of try execute
        //  BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(device));
        //  Make nonblocking and set flags
        //  Flags or decrement while finishing, tile_groups -> 0
        //  Caller provides pointer, set to tile groups in queue??
        //    Or poll how many tilegroups are still running?
        //
        BSG_CUDA_CALL(hb_mc_device_pod_try_launch_tile_groups(device, pod));

        eva_t recv_count_eva = count_device + CHAIN_LEN * sizeof(int);
        eva_t recv_buffer_eva = buffer_device + (CHAIN_LEN * BUFFER_ELS * sizeof(int));
        // NUM_PACKETS->BUFFER_ELS
        int packets_sent = 0;
        int count_host;
        void *src, *dst;

        bsg_manycore_spsc_queue_recv<int, BUFFER_ELS> recv_spsc(device, recv_buffer_eva, recv_count_eva);
        do
        {
            size_t xfer_sz = sizeof(int);
            eva_t buffer_eva = buffer_device + (packets_sent % BUFFER_ELS) * sizeof(int);
            hb_mc_npa_t buffer_npa;
            BSG_CUDA_CALL(hb_mc_eva_to_npa(mc, &default_map, &pod->mesh->origin, &buffer_eva, &buffer_npa, &xfer_sz));

            size_t count_sz = sizeof(int);
            eva_t count_eva = count_device;
            hb_mc_npa_t count_npa;
            BSG_CUDA_CALL(hb_mc_eva_to_npa(mc, &default_map, &pod->mesh->origin, &count_eva, &count_npa, &count_sz));
            if (packets_sent == 0)
            {
                printf("x86 BUFFER EVA/NPA: %x/%x\n", buffer_eva, buffer_npa);
            }
            
            src = (void *) ((intptr_t) count_eva);
            dst = (void *) &count_host;
            BSG_CUDA_CALL(hb_mc_device_memcpy(device, dst, src, sizeof(int), HB_MC_MEMCPY_TO_HOST));
            if (count_host < BUFFER_ELS)
            {
                dst = (void *) ((intptr_t) buffer_eva);
                src = (void *) &buffer_host[packets_sent];
                BSG_CUDA_CALL(hb_mc_device_memcpy(device, dst, src, sizeof(int), HB_MC_MEMCPY_TO_DEVICE));
                BSG_CUDA_CALL(hb_mc_manycore_host_request_fence(mc, -1));
                BSG_CUDA_CALL(hb_mc_manycore_amoadd(mc, &count_npa, 1, NULL));
                packets_sent++;
            }

            int recv_data;
            if (recv_spsc.try_recv(&recv_data))
            {
                printf("RECV-ing from buffer %d\n", recv_data);
            }

            // Write to core with broken reservation
            // Add interrupt on the host side 
            //   - response packet or request packet show up in fifo
            //   - interrupt handler that reads the packet off the fifo

            BSG_CUDA_CALL(hb_mc_device_pod_wait_for_tile_group_finish_any(device, pod, 10));
        } while (hb_mc_device_pod_all_tile_groups_finished(device, pod) != HB_MC_SUCCESS);
        


        /*****************************************************************************************************************
        * Freeze the tiles and memory manager cleanup. 
        ******************************************************************************************************************/
        BSG_CUDA_CALL(hb_mc_device_finish(device)); 

        int mismatch = 0; 
        for (int i = 0; i < NUM_PACKETS; i++) {
                if (pkt_data[i] != i) { 
                        bsg_pr_err(BSG_RED("Mismatch") ": -- A[%d] = 0x%08" PRIx32 "\t Expected: 0x%08" PRIx32 "\n", i , pkt_data[i], i);
                        mismatch = 1;
                }
        } 

        if (mismatch) { 
                return HB_MC_FAIL;
        }
        return HB_MC_SUCCESS;
}

declare_program_main("test_host_stream", kernel_host_stream);
