
/*******************************************************************************
 * (c) 2019 Paul Gordon's parallel (CUDA) NVIDIA GPU implementation of the Dynamic Time 
 * Warp Barycenter Averaging algorithm as conceived (without parallel compuation conception) by Francois Petitjean 
 ******************************************************************************/

#include "openDBA.cuh"

__host__
int main(int argc, char **argv){
	
	int norm_sequences = 0;
	
	char c;
	while( ( c = getopt (argc, argv, "n") ) != -1 ) {
		switch(c) {
			case 'n':
				norm_sequences = 1;
				break;
			default:
				/* You won't actually get here. */
				break;
		}
	}

	if(argc < 8){
#if HDF5_SUPPORTED == 1
		std::cout << "Usage: " << argv[0] << " <binary|text|tsv|fast5> ";
#else
		std::cout << "Usage: " << argv[0] << " <binary|text|tsv> "; 
#endif
#if DOUBLE_UNSUPPORTED == 1
		std::cout << "<int|uint|ulong|float> " <<
#else
		std::cout << "<int|uint|ulong|float|double> " <<
#endif
		          "<global|open_start|open_end|open> <output files prefix> <minimum unimodal segment length, or 0 for no segmentation> <prefix sequence to remove|/dev/null> <series.tsv|<series1> <series2> [series3...]>\n";
		exit(1);
     	}

	int num_series = argc-7;
	int min_segment_length = atoi(argv[5]); // reasonable settings for nanopore RNA dwell time distributions would be 4 (lower to 2 for DNA)
	int read_mode = TEXT_READ_MODE;
	if(!strcmp(argv[1],"binary")){
		read_mode = BINARY_READ_MODE;
	}
#if HDF5_SUPPORTED == 1
	else if(!strcmp(argv[1],"fast5")){
		read_mode = FAST5_READ_MODE;
	}
#endif
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

	char *seqprefix_filename = 0;
	if(strcmp(argv[6], "/dev/null")){
		seqprefix_filename = argv[6];
	}

	int argind = 7; // Where the file names start
	// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
	if(!strcmp(argv[2],"int")){
		setupAndRun<int>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
	}
	else if(!strcmp(argv[2],"uint")){
		setupAndRun<unsigned int>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
	}
	else if(!strcmp(argv[2],"ulong")){
		setupAndRun<unsigned long long>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
	}
	else if(!strcmp(argv[2],"float")){
		setupAndRun<float>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
	}
	// Only since CUDA 6.1 (Pascal and later architectures) is atomicAdd(double *...) supported.  Remove if you want to compile for earlier graphics cards.
#if DOUBLE_UNSUPPORTED == 1
#else
	else if(!strcmp(argv[2],"double")){
		setupAndRun<double>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
	}
#endif
	else{
		std::cerr << "Second argument (" << argv[2] << ") was not one of the accepted numerical representations: 'int', 'uint', 'ulong', 'float' or 'double'" << std::endl;
		exit(1);
	}

	// Following needed to allow cuda-memcheck to detect memory leaks
	cudaDeviceReset(); CUERR("Resetting GPU device");
}
