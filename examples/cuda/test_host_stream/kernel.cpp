//This kernel performs a barrier among all tiles in tile group 

#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"
#include "bsg_manycore_spsc_queue.hpp"
#include "bsg_tile_group_barrier.hpp"

bsg_barrier<bsg_tiles_X, bsg_tiles_Y> barrier;

#define BUFFER_ELS  10
#define CHAIN_LEN    4
#define NUM_PACKETS 100

extern "C" __attribute__ ((noinline))
int kernel_host_stream(int *buffer_chain, int *buffer_count)
{
    int *buffer = &buffer_chain[0] + (__bsg_id * BUFFER_ELS);
    int *count  = &buffer_count[0] + (__bsg_id);

    int *next_buffer = &buffer_chain[0] + ((__bsg_id+1) * BUFFER_ELS);
    int *next_count = &buffer_count[0] + (__bsg_id+1);

    bsg_printf("[%x] B %x C %x NB %x NC: %x\n", __bsg_id, buffer, count, next_buffer, next_count);

    bsg_manycore_spsc_queue_recv<int, BUFFER_ELS> recv_spsc(buffer, count);
    bsg_manycore_spsc_queue_send<int, BUFFER_ELS> send_spsc(next_buffer, next_count);

    int packets = 0;
    int recv_data;
    int send_data;
    do
    {
        recv_data = recv_spsc.recv();

        bsg_printf("[%d] RECV %d\n", __bsg_id, recv_data);

        send_data = recv_data;

        if (__bsg_id == CHAIN_LEN-1)
        {
            //int *ptr = (int*)bsg_remote_ptr_io(IO_X_INDEX, 0x8888);
            //*ptr = send_data;
            //bsg_printf("SENDING END OF CHAIN %d\n", send_data);
            send_spsc.send(send_data);
        }
        else
        {
            send_spsc.send(send_data);
        }

        bsg_printf("[%d] SEND %d (packet %d)\n", __bsg_id, send_data, packets);
    } while (++packets < NUM_PACKETS);

    if (__bsg_id == 0)
    {
        bsg_printf("syncing...\n");
    }
    barrier.sync();
    bsg_printf("[%d] finishing...\n", __bsg_id);
	return 0;
}

