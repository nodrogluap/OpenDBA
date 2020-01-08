
/*******************************************************************************
 * (c) 2019 Paul Gordon's parallel (CUDA) NVIDIA GPU implementation of the Dynamic Time 
 * Warp Barycenter Averaging algorithm as conceived (without parallel compuation conception) by Francois Petitjean 
 ******************************************************************************/

#include <string.h>
#include <iostream>
#include <fstream>
#include "cpu_utils.hpp"
#include "dba.hpp"
#include "segmentation.hpp"
#include "io_utils.hpp"

#define TEXT_READ_MODE 0
#define BINARY_READ_MODE 1
#define TSV_READ_MODE 2
#if HDF5_SUPPORTED == 1
#define FAST5_READ_MODE 3
#endif

template<typename T>
void
setupAndRun(char *seqprefix_file_name, char **series_file_names, int num_series, char *output_prefix, int read_mode, int use_open_start, int use_open_end, int min_segment_length){
	size_t *sequence_lengths = 0;
	size_t averageSequenceLength = 0;
	T **segmented_sequences = 0;
	size_t *segmented_seq_lengths = 0;
	void *averageSequence = 0;
	T **sequences = 0;
	int actual_num_series = 0; // excludes failed file reading

	// Step 0. Read in data.
        if(read_mode == BINARY_READ_MODE){ actual_num_series = readSequenceBinaryFiles<T>(series_file_names, num_series, &sequences, &sequence_lengths); }
	// In the following two the sequence names are from inside the file, not the file names themselves
        else if(read_mode == TSV_READ_MODE){ actual_num_series = readSequenceTSVFiles<T>(series_file_names, num_series, &sequences, &series_file_names, &sequence_lengths); }
#if HDF5_SUPPORTED == 1
        else if(read_mode == FAST5_READ_MODE){ 
		actual_num_series = readSequenceFAST5Files<T>(series_file_names, num_series, &sequences, &series_file_names, &sequence_lengths); 
#if DEBUG == 1
		writeSequences(sequences, sequence_lengths, series_file_names, actual_num_series, CONCAT2(output_prefix, ".seqs.txt").c_str());
#endif
	}
#endif
        else{ actual_num_series = readSequenceTextFiles<T>(series_file_names, num_series, &sequences, &sequence_lengths); }

	// Sanity check
	if(actual_num_series < 2){
		std::cerr << "At least two sequences must be provided to calculate an average, but found " << actual_num_series << ", aborting" << std::endl;
                exit(NOT_ENOUGH_SEQUENCES);
	}

	// Shorten sequence names to everything before the first "." in the file name
	for (int i = 0; i < actual_num_series; i++){ char *z = strchr(series_file_names[i], '.'); if(z) *z = '\0';}

	// Step 1. If a leading sequence was specified, chop it off all the inputs.
	if(seqprefix_file_name != 0){
		T **seqprefix = 0;
		size_t *seqprefix_length = 0;
		if(read_mode == BINARY_READ_MODE){
			readSequenceBinaryFiles<T>(&seqprefix_file_name, 1, &seqprefix, &seqprefix_length);
		}
		else{
			readSequenceTextFiles<T>(&seqprefix_file_name, 1, &seqprefix, &seqprefix_length);
		}
		if(*seqprefix_length == 0){
			std::cerr << "Cannot read prefix " << (read_mode == BINARY_READ_MODE ? "binary" : "text") << 
                                     " data from " << seqprefix_file_name << ", aborting" << std::endl;
			exit(CANNOT_READ_SEQUENCE_PREFIX_FILE);
		}
		chopPrefixFromSequences<T>(*seqprefix, *seqprefix_length, &sequences, actual_num_series, sequence_lengths, series_file_names, output_prefix);
		cudaFreeHost(*seqprefix); CUERR("Freeing CPU memory for the prefix sequence");
		cudaFreeHost(seqprefix); CUERR("Freeing CPU memory for the prefix sequencers pointer");
		cudaFreeHost(seqprefix_length); CUERR("Freeing CPU memory for the prefix sequence length");
	}
	// Step 2. If a minimum segment length was provided, segment the input sequences into unimodal pieces. 
	if(min_segment_length > 0){
		std::cout << "Segmenting with minimum acceptable segment size of " << min_segment_length << std::endl;
		adaptive_segmentation<T>(sequences, sequence_lengths, actual_num_series, min_segment_length, &segmented_sequences, &segmented_seq_lengths);
#if DEBUG == 1
		writeSequences(segmented_sequences, segmented_seq_lengths, series_file_names, actual_num_series, CONCAT2(output_prefix, ".segmented_seqs.txt").c_str());
#endif
        	for (int i = 0; i < actual_num_series; i++){ cudaFreeHost(sequences[i]); CUERR("Freeing CPU memory for a presegmentation sequence");}
        	cudaFreeHost(sequences); CUERR("Freeing CPU memory for the presegmentation sequence pointers");
		cudaFreeHost(sequence_lengths); CUERR("Freeing CPU memory for the presegmentation sequence lengths");
		sequences = segmented_sequences;
		sequence_lengths = segmented_seq_lengths;
	}

	// Step 3. The meat of this meal!
        performDBA<T>(sequences, actual_num_series, sequence_lengths, series_file_names, use_open_start, use_open_end, output_prefix, (T **) &averageSequence, &averageSequenceLength);

	// Step 4. Save results.
	std::ofstream avg_file(CONCAT2(output_prefix, ".avg.txt").c_str());
	if(!avg_file.is_open()){
		std::cerr << "Cannot open sequence averages file " << output_prefix << ".avg.txt for writing" << std::endl;
		exit(CANNOT_WRITE_DBA_AVG);
	}
        for (size_t i = 0; i < averageSequenceLength; ++i) { avg_file << ((T *) averageSequence)[i] << std::endl; }
	avg_file.close();

	// Cleanup
	for (int i = 0; i < actual_num_series; i++){ cudaFreeHost(sequences[i]); CUERR("Freeing CPU memory for a sequence");}
        cudaFreeHost(sequences); CUERR("Freeing CPU memory for the segmented sequence pointers");
	cudaFreeHost(sequence_lengths); CUERR("Freeing CPU memory for the segmented sequence lengths");
	cudaFreeHost(averageSequence); CUERR("Freeing CPU memory for the DBA result");
}

__host__
int main(int argc, char **argv){

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
		setupAndRun<int>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length);
	}
	else if(!strcmp(argv[2],"uint")){
		setupAndRun<unsigned int>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length);
	}
	else if(!strcmp(argv[2],"ulong")){
		setupAndRun<unsigned long long>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length);
	}
	else if(!strcmp(argv[2],"float")){
		setupAndRun<float>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length);
	}
	// Only since CUDA 6.1 (Pascal and later architectures) is atomicAdd(double *...) supported.  Remove if you want to compile for earlier graphics cards.
#if DOUBLE_UNSUPPORTED == 1
#else
	else if(!strcmp(argv[2],"double")){
		setupAndRun<double>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length);
	}
#endif
	else{
		std::cerr << "Second argument (" << argv[2] << ") was not one of the accepted numerical representations: 'int', 'uint', 'ulong', 'float' or 'double'" << std::endl;
		exit(1);
	}

	// Following needed to allow cuda-memcheck to detect memory leaks
	cudaDeviceReset(); CUERR("Resetting GPU device");
}
