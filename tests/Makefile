# #----General Definitions----#

# #----Compilers----#
# NVCC=nvcc

# #----Debug----#
# CPU_DEBUG=-g
# CUDA_DEBUG=-G

# #----Linux----#

# DOUBLE_UNSUPPORTED=0
# # By default do not require HDF5 to compile (enable to allow reading Oxford Nanopore Technologies FAST5 files)
# HDF5_SUPPORTED=0
# DEBUG=0

# #----Include Directories----#

# HDF5_FLAGS_LINUX=-lhdf5

# #----Linux Flags----#

# CXX11=-std=c++11
# C99=-std=c99

# #----make objects for linux----#


	
# openDBA_test: openDBA_test.cu ../openDBA.cu ../openDBA.cuh ../cpu_utils.hpp ../cuda_utils.hpp ../exit_codes.hpp ../dba.hpp ../multithreading.h ../multithreading.cpp ../gpu_utils.hpp ../io_utils.hpp ../dtw.hpp ../limits.hpp ../segmentation.hpp
	# $(NVCC) $(CXX11) $(CPU_DEBUG) $(CUDA_DEBUG) -DDEBUG=$(DEBUG) -DDOUBLE_UNSUPPORTED=$(DOUBLE_UNSUPPORTED) -DHDF5_SUPPORTED=$(HDF5_SUPPORTED) openDBA_test.cu ../openDBA.cu -o openDBA_test $(HDF5_FLAGS_LINUX)

#CXX=nvcc
# By default enable double precision float point number support (most NVIDIA GPU cards after 2016)
DOUBLE_UNSUPPORTED=0
# By default do not require HDF5 to compile (enable to allow reading Oxford Nanopore Technologies FAST5 files)
HDF5_SUPPORTED=0
DEBUG=1
# For kernel-side sqrt() support and getDeviceCount() calls respectively
NVCC_FLAGS+= --expt-relaxed-constexpr -rdc=true -maxrregcount 26 

ifeq ($(DEBUG),1)
  NVCC_FLAGS+= -g -G
endif

ifeq ($(HDF5_SUPPORTED),1)
  NVCC_FLAGS+= -lhdf5
endif

ifeq ($(DOUBLE_UNSUPPORTED),0)
  NVCC_FLAGS+= -arch=sm_61
endif

# Following two targets are small external libraries with more less restrictive licenses (see headers for license info)
multithreading.o: ../multithreading.cpp
	nvcc -c $< -o $@

fastcluster.o: ../fastcluster.cpp
	nvcc -DNO_INCLUDE_FENV -c $< -o $@ 

openDBA.o: openDBA_test.cu ../openDBA.cuh ../segmentation.hpp ../cpu_utils.hpp ../gpu_utils.hpp ../io_utils.hpp ../exit_codes.hpp ../dtw.hpp ../dba.hpp ../limits.hpp ../cuda_utils.hpp
	nvcc -std=c++11 -DDEBUG=$(DEBUG) -DDOUBLE_UNSUPPORTED=$(DOUBLE_UNSUPPORTED) -DHDF5_SUPPORTED=$(HDF5_SUPPORTED) $(NVCC_FLAGS) -c $< -o $@
	
openDBA_test: openDBA_test.cu test_utils.cuh Makefile openDBA.o multithreading.o fastcluster.o
	nvcc -std=c++11 $(NVCC_FLAGS) --compiler-options -fPIC openDBA.o multithreading.o fastcluster.o -o openDBA_test
