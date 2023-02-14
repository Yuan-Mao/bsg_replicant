# Copyright (c) 2019, University of Washington All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# Redistributions of source code must retain the above copyright notice, this list
# of conditions and the following disclaimer.
# 
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# 
# Neither the name of the copyright holder nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# aws-fpga and aws-vcs are identical, EXCEPT for the MMIO layer. The
# reuse the bsg_manycore_platform.cpp file between the two platforms
# aws-fpga, but provide our own bsg_manycore_mmio.cpp file that
# handles PCIE-based MMIO.
PLATFORM_CXXSOURCES += $(LIBRARIES_PATH)/platforms/zynqblade-fpga/bsg_manycore_platform.cpp
PLATFORM_CXXSOURCES += $(LIBRARIES_PATH)/features/profiler/noimpl/bsg_manycore_profiler.cpp
PLATFORM_CXXSOURCES += $(LIBRARIES_PATH)/features/tracer/noimpl/bsg_manycore_tracer.cpp

# aws-fpga does not provide a DMA feature. Therefore, we use the fragment in 
# features/dma/noimpl/feature.mk that simply returns
# HB_MC_NO_IMPL for each function call.
include $(LIBRARIES_PATH)/features/dma/noimpl/feature.mk

PLATFORM_OBJECTS += $(patsubst %cpp,%o,$(PLATFORM_CXXSOURCES))
PLATFORM_OBJECTS += $(patsubst %c,%o,$(PLATFORM_CSOURCES))

PLATFORM_REGRESSION_OBJECTS += $(patsubst %cpp,%o,$(PLATFORM_REGRESSION_CXXSOURCES))
PLATFORM_REGRESSION_OBJECTS += $(patsubst %c,%o,$(PLATFORM_REGRESSION_CSOURCES))

$(PLATFORM_OBJECTS) $(PLATFORM_REGRESSION_OBJECTS): INCLUDES := -I$(LIBRARIES_PATH)
$(PLATFORM_OBJECTS) $(PLATFORM_REGRESSION_OBJECTS): INCLUDES += -I$(LIBRARIES_PATH)/features/profiler
$(PLATFORM_OBJECTS) $(PLATFORM_REGRESSION_OBJECTS): INCLUDES += -I$(LIBRARIES_PATH)/features/tracer
$(PLATFORM_OBJECTS) $(PLATFORM_REGRESSION_OBJECTS): INCLUDES += -I$(BSG_PLATFORM_PATH)

$(PLATFORM_OBJECTS) $(PLATFORM_REGRESSION_OBJECTS): CFLAGS   := -std=c11 -fPIC -D_BSD_SOURCE -D_GNU_SOURCE -D_DEFAULT_SOURCE $(INCLUDES)
$(PLATFORM_OBJECTS) $(PLATFORM_REGRESSION_OBJECTS): CXXFLAGS := -std=c++11 -fPIC -D_BSD_SOURCE -D_GNU_SOURCE -D_DEFAULT_SOURCE $(INCLUDES)
$(PLATFORM_OBJECTS) $(PLATFORM_REGRESSION_OBJECTS): LDFLAGS   = -fPIC
$(PLATFORM_REGRESSION_OBJECTS): LDFLAGS   = -ldl

$(BSG_PLATFORM_PATH)/libbsg_manycore_runtime.so.1.0: $(PLATFORM_OBJECTS)
$(BSG_PLATFORM_PATH)/libbsg_manycore_regression.so.1.0: $(PLATFORM_REGRESSION_OBJECTS)

# Mirror the extensions linux installation in /usr/lib provides so
# that we can use -lbsg_manycore_runtime
$(BSG_PLATFORM_PATH)/libbsg_manycore_runtime.so.1: %: %.0
	ln -sf $@.0 $@

$(BSG_PLATFORM_PATH)/libbsgmc_cuda_legacy_pod_repl.so.1: %: %.0
	ln -sf $@.0 $@

$(BSG_PLATFORM_PATH)/libbsg_manycore_regression.so.1: %: %.0
	ln -sf $@.0 $@

$(BSG_PLATFORM_PATH)/libbsg_manycore_runtime.so: %: %.1
	ln -sf $@.1 $@

$(BSG_PLATFORM_PATH)/libbsgmc_cuda_legacy_pod_repl.so: %: %.1
	ln -sf $@.1 $@

$(BSG_PLATFORM_PATH)/libbsg_manycore_regression.so: %: %.1
	ln -sf $@.1 $@

.PHONY: platform.clean install uninstall
platform.clean:
	rm -f $(PLATFORM_OBJECTS)

libraries.clean: platform.clean

