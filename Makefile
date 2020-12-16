PROGNAME=openDBA
#CXX=nvcc
# By default enable double precision float point number support (most NVIDIA GPU cards after 2016)
DOUBLE_UNSUPPORTED=0
# By default do not require HDF5 to compile (enable to allow reading Oxford Nanopore Technologies FAST5 files)
HDF5_SUPPORTED=0
DEBUG=0
# For kernel-side sqrt() support and getDeviceCount() calls respectively
NVCC_FLAGS+= --expt-relaxed-constexpr -rdc=true -maxrregcount 26 --std=c++11

ifeq ($(DEBUG),1)
  NVCC_FLAGS+= -g -G
endif

ifeq ($(HDF5_SUPPORTED),1)
  NVCC_FLAGS+= -lhdf5
endif

ifeq ($(DOUBLE_UNSUPPORTED),0)
  NVCC_FLAGS+= -arch=sm_61
endif

all: $(PROGNAME)

clean:
	rm -f openDBA.o multithreading.o fastcluster.o $(PROGNAME)

# Following two targets are small external libraries with more less restrictive licenses (see headers for license info)
multithreading.o: multithreading.cpp
	nvcc -c $< -o $@

fastcluster.o: fastcluster.cpp
	nvcc -DNO_INCLUDE_FENV -c $< -o $@ 

openDBA.o: openDBA.cu openDBA.cuh segmentation.hpp cpu_utils.hpp gpu_utils.hpp io_utils.hpp exit_codes.hpp dtw.hpp dba.hpp limits.hpp cuda_utils.hpp
	nvcc -DDEBUG=$(DEBUG) -DDOUBLE_UNSUPPORTED=$(DOUBLE_UNSUPPORTED) -DHDF5_SUPPORTED=$(HDF5_SUPPORTED) $(NVCC_FLAGS) -c $< -o $@

$(PROGNAME): Makefile openDBA.o multithreading.o fastcluster.o
	nvcc $(NVCC_FLAGS) --compiler-options -fPIC openDBA.o multithreading.o fastcluster.o -o $(PROGNAME)

