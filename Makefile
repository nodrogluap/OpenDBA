CXX=nvcc

all: DBA

DBA: openDBA.cu multithreading.o cpu_utils.hpp gpu_utils.hpp dtw.hpp dba.hpp limits.hpp cuda_utils.hpp
	nvcc -arch=sm_61 --expt-relaxed-constexpr openDBA.cu multithreading.o -o openDBA
