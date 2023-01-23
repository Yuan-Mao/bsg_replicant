
#ifndef _BSG_MANYCORE_SPSC_QUEUE_HPP
#define _BSG_MANYCORE_SPSC_QUEUE_HPP

template <typename T, int S>
class bsg_manycore_spsc_queue_recv {
private:
    hb_mc_device_t *device;
    hb_mc_manycore_t *mc;
    hb_mc_eva_t buffer_eva;
    hb_mc_eva_t count_eva;
    hb_mc_npa_t buffer_npa;
    hb_mc_npa_t count_npa;
    T rptr;
    T count;

public:
    bsg_manycore_spsc_queue_recv(hb_mc_device_t *device, eva_t buffer_eva, eva_t count_eva) 
        : device(device), mc(device->mc), buffer_eva(buffer_eva), count_eva(count_eva), rptr(0)
    {
        hb_mc_pod_id_t pod_id = device->default_pod_id;
        hb_mc_pod_t *pod = &device->pods[pod_id];
        size_t xfer_sz = sizeof(T);
        // TODO: Check return and error?
        hb_mc_eva_to_npa(mc, &default_map, &pod->mesh->origin, &buffer_eva, &buffer_npa, &xfer_sz);
        hb_mc_eva_to_npa(mc, &default_map, &pod->mesh->origin, &count_eva, &count_npa, &xfer_sz);

        printf("Buffer EVA %x Count EVA %x Buffer NPA %x Count NPA %x\n",
                buffer_eva, count_eva, buffer_npa, count_npa);
    };

    bool is_empty(void)
    {
        void *src = (void *) ((intptr_t) count_eva);
        void *dst = (void *) &count;
        BSG_CUDA_CALL(hb_mc_device_memcpy(device, dst, src, sizeof(T), HB_MC_MEMCPY_TO_HOST));
        return count == 0;
    }

    bool try_recv(T *data)
    {
        if (is_empty())
        {
            return false;
        }

        void *dst = (void *) ((intptr_t) buffer_eva+rptr*sizeof(T));
        void *src = (void *) data;
        BSG_CUDA_CALL(hb_mc_device_memcpy(device, dst, src, sizeof(T), HB_MC_MEMCPY_TO_DEVICE));
        BSG_CUDA_CALL(hb_mc_manycore_host_request_fence(mc, -1));
        // Probably faster than modulo, but should see if compiler
        //   optimizes...
        if (++rptr == S)
        {
            rptr = 0;
        }
        BSG_CUDA_CALL(hb_mc_manycore_amoadd(mc, &count_npa, -1, NULL));

        return true;
    }

    // TODO: Add timeout?
    T recv(void)
    {
        T data;
        while (1)
        {
            if (try_recv(&data)) break;
        }

        return data;
    }
};

//template <typename T, int S>
//class bsg_manycore_spsc_queue_send {
//private:
//    volatile T *buffer;
//    volatile int *count;
//    int wptr;
//
//public:
//    bsg_manycore_spsc_queue_send(T *buffer, int *count) 
//        : buffer(buffer), count(count), wptr(0) { };
//
//    bool is_full(void)
//    {
//        return (*count == S);
//    }
//
//    bool try_send(T data)
//    {
//        if (is_full()) return false;
//
//        buffer[wptr] = data;
//        // Probably faster than modulo, but should see if compiler
//        //   optimizes...
//        if (++wptr == S)
//        {
//            wptr = 0;
//        }
//        bsg_fence();
//        bsg_amoadd(count, 1);
//
//        return true;
//    }
//
//    // TODO: Add timeout?
//    void send(T data)
//    {
//        while (1)
//        {
//            if (try_send(data)) break;
//        }
//    }
//};

#endif

