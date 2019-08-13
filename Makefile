PROGNAME=openDBA
CXX=nvcc
# By default enable double precisuion float point number support (most NVIDIA GPU cards after 2016)
DOUBLE_UNSUPPORTED=0
# For kernel-side sqrt() support
NVCC_FLAGS+= --expt-relaxed-constexpr

ifeq ($(DOUBLE_UNSUPPORTED),0)
  NVCC_FLAGS+= -arch=sm_61
endif

all: $(PROGNAME)

$(PROGNAME): openDBA.cu multithreading.o cpu_utils.hpp gpu_utils.hpp dtw.hpp dba.hpp limits.hpp cuda_utils.hpp
	nvcc -DDOUBLE_UNSUPPORTED=$(DOUBLE_UNSUPPORTED) $(NVCC_FLAGS) openDBA.cu multithreading.o -o $(PROGNAME)
