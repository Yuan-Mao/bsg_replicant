#include <bsg_manycore_config.h>

const char *hb_mc_config_id_to_string(hb_mc_config_id_t id)
{
    static const char *strtab [] = {
        [HB_MC_CONFIG_VERSION] = "BLADERUNNER HARDWARE VERSION",
        [HB_MC_CONFIG_TIMESTAMP] = "BLADERUNNER COMPILATION DATE TIMESTAMP",
        [HB_MC_CONFIG_NETWORK_ADDR_WIDTH] = "BLADERUNNER NETWORK ADDRESS WDITH",
        [HB_MC_CONFIG_NETWORK_DATA_WIDTH] = "BLADERUNNER NETWORK DATA WIDTH",
        [HB_MC_CONFIG_POD_DIM_X] = "BLADERUNNER POD DIMENSION X",
        [HB_MC_CONFIG_POD_DIM_Y] = "BLADERUNNER POD DIMENSION Y",
        [HB_MC_CONFIG_DIM_PODS_X] = "BLADERUNNER NUMBER OF PODS X",
        [HB_MC_CONFIG_DIM_PODS_Y] = "BLADERUNNER NUMBER OF PODS Y",
        [HB_MC_CONFIG_DEVICE_HOST_INTF_COORD_X] = "BLADERUNNER HOST INTERFACE DIMENSION X",
        [HB_MC_CONFIG_DEVICE_HOST_INTF_COORD_Y] = "BLADERUNNER HOST INTERFACE DIMENSION Y",
        [HB_MC_CONFIG_NOC_COORD_X_WIDTH] = "BLADERUNNER NOC COORD X WIDTH",
        [HB_MC_CONFIG_NOC_COORD_Y_WIDTH] = "BLADERUNNER NOC COORD Y WIDTH",
        [HB_MC_CONFIG_NOC_RUCHE_FACTOR_X] = "BLADERUNNER NOC RUCHE FACTOR X",
        [HB_MC_CONFIG_BARRIER_RUCHE_FACTOR_X] = "BLADERUNNER BARRIER RUCHE FACTOR X",
        [HB_MC_CONFIG_WH_RUCHE_FACTOR_X] = "BLADERUNNER WH RUCHE FACTOR X",
        [HB_MC_CONFIG_REPO_BASEJUMP_HASH] = "BLADERUNNER REPO BASEJUMP HASH",
        [HB_MC_CONFIG_REPO_MANYCORE_HASH] = "BLADERUNNER REPO MANYCORE HASH",
        [HB_MC_CONFIG_REPO_F1_HASH] = "BLADERUNNER REPO F1 HASH",
        [HB_MC_CONFIG_VCACHE_WAYS]  = "BLADERUNNER VCACHE WAYS",
        [HB_MC_CONFIG_VCACHE_SETS]  = "BLADERUNNER VCACHE SETS",
        [HB_MC_CONFIG_VCACHE_BLOCK_WORDS] = "BLADERUNNER VCACHE BLOCK SIZE IN WORDS",
        [HB_MC_CONFIG_VCACHE_STRIPE_WORDS] = "BLADERUNNER VCACHE STRIPE SIZE IN WORDS",
        [HB_MC_CONFIG_VCACHE_MISS_FIFO_ELS] = "BLADERUNNER VCACHE MISS FIFO ELS",
        [HB_MC_CONFIG_IO_REMOTE_LOAD_CAP] = "BLADERUNNER IO REMOTE LOAD CAPACITY",
        [HB_MC_CONFIG_IO_HOST_CREDITS_CAP] = "BLADERUNNER IO HOST REQUEST CREDITS CAPACITY",
        [HB_MC_CONFIG_IO_EP_MAX_OUT_CREDITS] = "BLADERUNNER IO ENDPOINT MAX OUT CREDITS",
    };
    return strtab[id];
}

