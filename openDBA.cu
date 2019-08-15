
/*******************************************************************************
 * (c) 2019 Paul Gordon's parallel (CUDA) NVIDIA GPU implementation of the Dynamic Time 
 * Warp Barycenter Averaging algorithm as conceived (without parallel compuation conception) by Francois Petitjean 
 ******************************************************************************/

#include <string.h>
#include <iostream>
#include <fstream>
#include "cpu_utils.hpp"
#include "dba.hpp"

#define TEXT_READ_MODE 0
#define BINARY_READ_MODE 1
#define TSV_READ_MODE 2

template<typename T>
void
setupAndRun(char **series_file_names, int num_series, char *output_prefix, int read_mode, int use_open_start, int use_open_end, double convergence_delta){
	size_t *sequence_lengths = 0;
	size_t averageSequenceLength = 0;
	void *averageSequence = 0;
	T **sequences = 0;
	int actual_num_series = 0; // excludes failed file reading
        if(read_mode == BINARY_READ_MODE){ actual_num_series = readSequenceBinaryFiles<T>(series_file_names, num_series, &sequences, &sequence_lengths); }
        else if(read_mode == TSV_READ_MODE){ actual_num_series = readSequenceTSVFiles<T>(series_file_names, num_series, &sequences, &series_file_names, &sequence_lengths); }
        else{ actual_num_series = readSequenceTextFiles<T>(series_file_names, num_series, &sequences, &sequence_lengths); }

	// Shorten sequence names to everything before the first "." in the file name
	for (int i = 0; i < actual_num_series; i++){ char *z = strchr(series_file_names[i], '.'); if(z) *z = '\0';}
        performDBA<T>(sequences, actual_num_series, sequence_lengths, series_file_names, convergence_delta, use_open_start, use_open_end, output_prefix, (T **) &averageSequence, &averageSequenceLength);

	std::ofstream avg_file((std::string(output_prefix)+std::string(".avg.txt")).c_str());
	if(!avg_file.is_open()){
		std::cerr << "Cannot open sequence averages file for writing" << std::endl;
		exit(3);
	}
        for (size_t i = 0; i < averageSequenceLength; ++i) { avg_file << ((T *) averageSequence)[i] << std::endl; }
	avg_file.close();

	// Cleanup
        for (int i = 0; i < num_series; i++){ cudaFreeHost(sequences[i]); }
        cudaFreeHost(sequences);
	cudaFreeHost(sequence_lengths); CUERR("Freeing CPU memory for the sequence lengths");
	cudaFreeHost(averageSequence); CUERR("Freeing CPU memory for the DBA result");
}

__host__
int main(int argc, char **argv){

	if(argc < 7){
#if DOUBLE_UNSUPPORTED == 1
		std::cout << "Usage: " << argv[0] << " <binary|text|tsv> <int|uint|ulong|float> " <<
#else
		std::cout << "Usage: " << argv[0] << " <binary|text|tsv> <int|uint|ulong|float|double> " <<
#endif
		          "<global|open_start|open_end|open> <output files prefix> <delta criterium for convergence, in range (0,1]> <series.tsv|<series1> <series2> [series3...]>\n";
		exit(1);
     	}

	int num_series = argc-6;
	double convergence_delta = atof(argv[5]);
	if(convergence_delta <= 0.0 || convergence_delta > 1){
		std::cerr << "Fifth argument (" << argv[3] << ") could not be parsed into a number in the acceptable range (0,1]" << std::endl;
		exit(1);
	} 
	int read_mode = TEXT_READ_MODE;
	if(!strcmp(argv[1],"binary")){
		read_mode = BINARY_READ_MODE;
	}
	else if(!strcmp(argv[1],"tsv")){
		read_mode = TSV_READ_MODE;
	}
	else if(strcmp(argv[1],"text")){
		std::cerr << "First argument (" << argv[1] << ") is neither 'binary' nor 'text'" << std::endl;
		exit(1);
	}

	int use_open_start = 0;
	int use_open_end = 0;
	if(!strcmp(argv[3],"global")){
        }
        else if(!strcmp(argv[3],"open_start")){
		use_open_start = 1;
        }
        else if(!strcmp(argv[3],"open_end")){
		use_open_end = 1;
        }
	else if(!strcmp(argv[3],"open")){
		use_open_start = 1;
		use_open_end = 1;
        }
	else{
		std::cerr << "Third argument (" << argv[3] << ") is not one of the accept values 'global', 'open_start', 'open_end' or 'open'" << std::endl;
                exit(1);
	}

	char *output_prefix = argv[4];

	int argind = 6; // Where the file names start
	// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
	if(!strcmp(argv[2],"int")){
		setupAndRun<int>(&argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, convergence_delta);
	}
	else if(!strcmp(argv[2],"uint")){
		setupAndRun<unsigned int>(&argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, convergence_delta);
	}
	else if(!strcmp(argv[2],"ulong")){
		setupAndRun<unsigned long long>(&argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, convergence_delta);
	}
	else if(!strcmp(argv[2],"float")){
		setupAndRun<float>(&argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, convergence_delta);
	}
	// Only since CUDA 6.1 (Pascal and later architectures) is atomicAdd(double *...) supported.  Remove if you want to compile for earlier graphics cards.
#if DOUBLE_UNSUPPORTED == 1
#else
	else if(!strcmp(argv[2],"double")){
		setupAndRun<double>(&argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, convergence_delta);
	}
#endif
	else{
		std::cerr << "Second argument (" << argv[2] << ") was not one of the accepted numerical representations: 'int', 'uint', 'ulong', 'float' or 'double'" << std::endl;
		exit(1);
	}

	// Following needed to allow cuda-memcheck to detect memory leaks
	cudaDeviceReset(); CUERR("Resetting GPU device");
}
