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

# This Makefile fragment defines all of the rules for linking
# aws-vcs binaries

ORANGE=\033[0;33m
RED=\033[0;31m
NC=\033[0m

# This file REQUIRES several variables to be set. They are typically set by the
# Makefile that includes this fragment...

# BSG_PLATFORM_PATH: The path to the testbenches folder in BSG F1
ifndef BSG_PLATFORM_PATH
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: BSG_PLATFORM_PATH is not defined$(NC)"))
endif

# hardware.mk is the file list for the simulation RTL. The one
# hardware/hardware.mk includes the platform specific file,
# aws-vcs/hardware.mk. aws-vcs/hardware.mk defines how to build the simulation
# objects for VCS: $(BSG_PLATFORM_PATH)/vcs_simlibs/$(BSG_MACHINE_NAME)/AN.DB
include $(HARDWARE_PATH)/hardware.mk

# libraries.mk defines how to build libbsg_manycore_runtime.so, which is
# pre-linked against all other simulation binaries. 
include $(LIBRARIES_PATH)/libraries.mk

# msg_config is generated by the msg_config.mk file. It reduces verbosity of
# VCS, but that's a low bar...
include $(BSG_PLATFORM_PATH)/msg_config.mk

# libbsg_manycore_runtime will be compiled in $(BSG_PLATFORM_PATH)
LDFLAGS    += -lbsg_manycore_regression -lbsg_manycore_runtime -lm
LDFLAGS    += -L$(BSG_PLATFORM_PATH) -Wl,-rpath=$(BSG_PLATFORM_PATH)

VCS_LDFLAGS    += $(foreach def,$(LDFLAGS),-LDFLAGS "$(def)")
VCS_VFLAGS     += -M -L -ntb_opts tb_timescale=1ps/1ps -lca -v2005
VCS_VFLAGS     += -timescale=1ps/1ps -sverilog -full64 -licqueue -q
VCS_VFLAGS     += +warn=noLCA_FEATURES_ENABLED
VCS_VFLAGS     += +warn=noMC-FCNAFTMI
VCS_VFLAGS     += +lint=all,TFIPC-L,noSVA-UA,noSVA-NSVU,noVCDE,noSVA-AECASR
VCS_VFLAGS     += -msg_config=$(BSG_PLATFORM_PATH)/msg_config

# VCS Generates an executable file by linking the %.o file with the
# the VCS work libraries for the design, and the runtime shared libraries

# The % and %.debug rules are identical. They must be separate
# otherwise make doesn't match the latter target(s)
%: %.o $(BSG_PLATFORM_PATH)/vcs_simlibs/$(BSG_MACHINE_NAME)/AN.DB $(BSG_PLATFORM_PATH)/libbsg_manycore_regression.so $(BSG_PLATFORM_PATH)/libbsg_manycore_runtime.so $(BSG_PLATFORM_PATH)/msg_config
	SYNOPSYS_SIM_SETUP=$(BSG_MACHINE_PATH)/synopsys_sim.setup \
	vcs $(BSG_MACHINE_NAME) tb glbl $< $(VCS_LDFLAGS) $(VCS_VFLAGS) \
		-Mdirectory=$@.tmp -o $@ -l $@.vcs.log

%.debug: VCS_VFLAGS += -debug_pp 
%.debug: VCS_VFLAGS += +plusarg_save +vcs+vcdpluson +vcs+vcdplusmemon +memcbk
%.debug: %.o $(BSG_PLATFORM_PATH)/vcs_simlibs/$(BSG_MACHINE_NAME)/AN.DB $(BSG_PLATFORM_PATH)/libbsg_manycore_regression.so $(BSG_PLATFORM_PATH)/libbsg_manycore_runtime.so $(BSG_PLATFORM_PATH)/msg_config
	SYNOPSYS_SIM_SETUP=$(BSG_MACHINE_PATH)/synopsys_sim.setup \
	vcs $(BSG_MACHINE_NAME) tb glbl $< $(VCS_LDFLAGS) $(VCS_VFLAGS) \
		-Mdirectory=$@.tmp -o $@ -l $@.vcs.log

.PHONY: platform.link.clean
platform.link.clean:
	rm -rf *.daidir *.tmp
	rm -rf *.jou
	rm -rf *.vcs.log
	rm -rf vc_hdrs.h
	rm -rf $(BSG_PLATFORM_PATH)/msg_config
	rm -rf *.debug

link.clean: platform.link.clean ;