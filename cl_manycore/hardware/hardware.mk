include $(CL_DIR)/Makefile.machine.include
BSG_IP_CORES_COMMIT_ID ?= deadbeef
BSG_MANYCORE_COMMIT_ID ?= feedcafe
BSG_F1_COMMIT_ID       ?= 42c0ffee
FPGA_IMAGE_VERSION     ?= 0.0.0
CL_MANYCORE_MAX_EPA_WIDTH            := $(BSG_MACHINE_MAX_EPA_WIDTH)
CL_MANYCORE_DATA_WIDTH               := $(BSG_MACHINE_DATA_WIDTH)
CL_MANYCORE_VCACHE_WAYS              := $(BSG_MACHINE_VCACHE_WAY)
CL_MANYCORE_VCACHE_SETS              := $(BSG_MACHINE_VCACHE_SET)
CL_MANYCORE_VCACHE_BLOCK_SIZE_WORDS  := $(BSG_MACHINE_VCACHE_BLOCK_SIZE_WORDS)
CL_MANYCORE_VCACHE_STRIPE_SIZE_WORDS := $(BSG_MACHINE_VCACHE_STRIPE_SIZE_WORDS)

CL_MANYCORE_RELEASE_VERSION          ?= $(shell echo $(FPGA_IMAGE_VERSION) | sed 's/\([0-9]*\)\.\([0-9]*\).\([0-9]*\)/000\10\20\3/')
CL_MANYCORE_COMPILATION_DATE         ?= $(shell date +%m%d%Y)

# The manycore architecture sources are defined in arch_filelist.mk. The
# unsynthesizable simulation sources (for tracing, etc) are defined in
# sim_filelist.mk. Each file adds to VSOURCES and VINCLUDES and depends on
# BSG_MANYCORE_DIR
include $(BSG_MANYCORE_DIR)/machines/arch_filelist.mk

# So that we can limit tool-specific to a few specific spots we use VDEFINES,
# VINCLUDES, and VSOURCES to hold lists of macros, include directores, and
# verilog sources (respectively). These are used during simulation compilation,
# but transformed into a tool-specific syntax where necesssary.
VINCLUDES += $(HARDWARE_PATH)

VSOURCES += $(HARDWARE_PATH)/bsg_bladerunner_configuration.v
VSOURCES += $(HARDWARE_PATH)/cl_manycore_pkg.v
VSOURCES += $(HARDWARE_PATH)/cl_manycore.sv
VSOURCES += $(HARDWARE_PATH)/bsg_manycore_wrapper.v

VSOURCES += $(CL_DIR)/../hdl/bsg_bladerunner_rom.v
VSOURCES += $(CL_DIR)/../hdl/axil_to_mcl.v
VSOURCES += $(CL_DIR)/../hdl/s_axil_mcl_adapter.v
VSOURCES += $(CL_DIR)/../hdl/axil_to_mem.sv

VHEADERS += $(HARDWARE_PATH)/f1_parameters.vh

CLEANS += hardware.clean

$(HARDWARE_PATH)/bsg_bladerunner_configuration.rom: $(CL_DIR)/Makefile.machine.include
	python $(HARDWARE_PATH)/create_bladerunner_rom.py \
                --comp-date=$(CL_MANYCORE_COMPILATION_DATE) \
                --network-addr-width=$(CL_MANYCORE_MAX_EPA_WIDTH) \
                --network-data-width=$(CL_MANYCORE_DATA_WIDTH) \
                --network-x=$(CL_MANYCORE_DIM_X) \
                --network-y=$(CL_MANYCORE_DIM_Y) \
                --host-coord-x=$(CL_MANYCORE_HOST_COORD_X) \
                --host-coord-y=$(CL_MANYCORE_HOST_COORD_Y) \
                --mc-version=$(CL_MANYCORE_RELEASE_VERSION) \
                bsg_ip_cores@$(BSG_IP_CORES_COMMIT_ID) \
                bsg_manycore@$(BSG_MANYCORE_COMMIT_ID) \
                bsg_f1@$(BSG_F1_COMMIT_ID) > $@

$(HARDWARE_PATH)/%.v: $(HARDWARE_PATH)/%.rom
	python $(BASEJUMP_STL_DIR)/bsg_mem/bsg_ascii_to_rom.py $< \
               bsg_bladerunner_configuration > $@

$(HARDWARE_PATH)/f1_parameters.vh: $(CL_DIR)/Makefile.machine.include
	@echo "\`ifndef F1_DEFINES" > $@
	@echo "\`define F1_DEFINES" >> $@
	@echo "\`define CL_MANYCORE_MAX_EPA_WIDTH $(CL_MANYCORE_MAX_EPA_WIDTH)" >> $@
	@echo "\`define CL_MANYCORE_DATA_WIDTH $(CL_MANYCORE_DATA_WIDTH)" >> $@
	@echo "\`define CL_MANYCORE_DIM_X $(CL_MANYCORE_DIM_X)" >> $@
	@echo "\`define CL_MANYCORE_DIM_Y $(CL_MANYCORE_DIM_Y)" >> $@
	@echo "\`define CL_MANYCORE_VCACHE_SETS $(CL_MANYCORE_VCACHE_SETS)" >> $@
	@echo "\`define CL_MANYCORE_VCACHE_WAYS $(CL_MANYCORE_VCACHE_WAYS)" >> $@
	@echo "\`define CL_MANYCORE_VCACHE_BLOCK_SIZE_WORDS $(CL_MANYCORE_VCACHE_BLOCK_SIZE_WORDS)" >> $@
	@echo "\`define CL_MANYCORE_VCACHE_STRIPE_SIZE_WORDS $(CL_MANYCORE_VCACHE_STRIPE_SIZE_WORDS)" >> $@
	@echo "\`define CL_MANYCORE_MEM_CFG $(CL_MANYCORE_MEM_CFG)" >> $@
	@echo "\`endif" >> $@

.PHONY: hardware.clean

hardware.clean:
	rm -f $(HARDWARE_PATH)/bsg_bladerunner_configuration.{rom,v}
	rm -f $(HARDWARE_PATH)/f1_parameters.vh


