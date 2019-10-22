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

INCLUDES += -I$(LIBRARIES_PATH)

CSOURCES   += 
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_bits.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_config.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_cuda.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_elf.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_eva.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_loader.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_memory_manager.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_origin_eva_map.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_print_int_responder.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_printing.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_request_packet_id.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_responder.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_tile.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_uart_responder.cpp
CXXSOURCES += $(LIBRARIES_PATH)/bsg_manycore_trace_responder.cpp

HEADERS += $(LIBRARIES_PATH)/bsg_manycore.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_bits.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_config.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_cuda.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_elf.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_eva.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_loader.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_memory_manager.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_origin_eva_map.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_printing.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_request_packet_id.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_responder.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_tile.h

HEADERS += $(LIBRARIES_PATH)/bsg_manycore_vcache.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_mmio.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_errno.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_features.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_coordinate.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_npa.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_epa.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_packet.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_response_packet.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_request_packet.h
HEADERS += $(LIBRARIES_PATH)/bsg_manycore_fifo.h

CHEADERS   += $(HEADERS)
CXXHEADERS += $(HEADERS)

OBJECTS += $(patsubst %cpp,%o,$(CXXSOURCES))
OBJECTS += $(patsubst %c,%o,$(CSOURCES))

$(OBJECTS): INCLUDES  = -I$(LIBRARIES_PATH)
$(OBJECTS): INCLUDES += -I$(SDK_DIR)/userspace/include
$(OBJECTS): INCLUDES += -I$(HDK_DIR)/common/software/include
$(OBJECTS): INCLUDES += -I$(AWS_FPGA_REPO_DIR)/SDAccel/userspace/include
$(OBJECTS): CFLAGS    = -std=c11 -fPIC -D_GNU_SOURCE $(INCLUDES)
$(OBJECTS): CXXFLAGS  = -std=c++11 -fPIC -D_GNU_SOURCE $(INCLUDES)
$(OBJECTS): LDFLAGS   = -lfpga_mgmt -fPIC
$(OBJECTS): $(HEADERS)

# Objects that should be compiled with debug flags
DEBUG_OBJECTS  +=
#DEBUG_OBJECTS  += $(LIBRARIES_PATH)/bsg_manycore_responder.o
#DEBUG_OBJECTS  += $(LIBRARIES_PATH)/bsg_manycore_uart_responder.o
$(DEBUG_OBJECTS):  CXXFLAGS += -DDEBUG

# Objects that should be compiled with strict compilation flags
STRICT_OBJECTS +=
STRICT_OBJECTS += $(LIBRARIES_PATH)/bsg_manycore_responder.o
STRICT_OBJECTS += $(LIBRARIES_PATH)/bsg_manycore_loader.o
STRICT_OBJECTS += $(LIBRARIES_PATH)/bsg_manycore_packet_id.o
STRICT_OBJECTS += $(LIBRARIES_PATH)/bsg_manycore_eva.o
STRICT_OBJECTS += $(LIBRARIES_PATH)/bsg_manycore_origin_eva_map.o
STRICT_OBJECTS += $(LIBRARIES_PATH)/bsg_manycore_print_int_responder.o
$(STRICT_OBJECTS): CXXFLAGS += -Wall -Werror
$(STRICT_OBJECTS): CXXFLAGS += -Wno-unused-variable
$(STRICT_OBJECTS): CXXFLAGS += -Wno-unused-function
$(STRICT_OBJECTS): CXXFLAGS += -Wno-unused-but-set-variable

$(LIBRARIES_PATH)/libbsg_manycore_runtime.so.1.0: LD = $(CXX)
$(LIBRARIES_PATH)/libbsg_manycore_runtime.so.1.0: LDFLAGS = -lfpga_mgmt -fPIC
$(LIBRARIES_PATH)/libbsg_manycore_runtime.so.1.0: $(OBJECTS)
	$(LD) -shared -Wl,-soname,$(basename $(notdir $@)) -o $@ $^ $(LDFLAGS)

.PHONY: libraries.clean
libraries.clean:
	rm -f $(LIBRARIES_PATH)/*.o
	rm -f $(LIBRARIES_PATH)/libbsg_manycore_runtime.so.1.0