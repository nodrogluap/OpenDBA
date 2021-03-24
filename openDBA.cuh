#ifndef OPENDBA_H
#define OPENDBA_H

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
setupAndRun(char *seqprefix_file_name, char **series_file_names, int num_series, char *output_prefix, int read_mode, int use_open_start, int use_open_end, int min_segment_length, int norm_sequences);

template<typename T>
void
setupAndRun(char *seqprefix_file_name, char **series_file_names, int num_series, char *output_prefix, int read_mode, int use_open_start, int use_open_end, int min_segment_length, int norm_sequences){
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
		writeSequences(sequences, sequence_lengths, series_file_names, actual_num_series, CONCAT2(output_prefix, ".seqs.txt").c_str());
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
		chopPrefixFromSequences<T>(*seqprefix, *seqprefix_length, &sequences, &actual_num_series, sequence_lengths, series_file_names, output_prefix, norm_sequences);
		cudaFreeHost(*seqprefix); CUERR("Freeing CPU memory for the prefix sequence");
		cudaFreeHost(seqprefix); CUERR("Freeing CPU memory for the prefix sequencers pointer");
		cudaFreeHost(seqprefix_length); CUERR("Freeing CPU memory for the prefix sequence length");
	}
	// Step 2. If a minimum segment length was provided, segment the input sequences into unimodal pieces. 
	if(min_segment_length > 0){
		std::cout << "Segmenting with minimum acceptable segment size of " << min_segment_length << std::endl;
		adaptive_segmentation<T>(sequences, sequence_lengths, actual_num_series, min_segment_length, &segmented_sequences, &segmented_seq_lengths);
		writeSequences(segmented_sequences, segmented_seq_lengths, series_file_names, actual_num_series, CONCAT2(output_prefix, ".segmented_seqs.txt").c_str());
		for (int i = 0; i < actual_num_series; i++){ cudaFreeHost(sequences[i]); CUERR("Freeing CPU memory for a presegmentation sequence");}
		cudaFreeHost(sequences); CUERR("Freeing CPU memory for the presegmentation sequence pointers");
		cudaFreeHost(sequence_lengths); CUERR("Freeing CPU memory for the presegmentation sequence lengths");
		sequences = segmented_sequences;
		sequence_lengths = segmented_seq_lengths;
	}

	// Step 3. The meat of this meal!
	performDBA<T>(sequences, actual_num_series, sequence_lengths, series_file_names, use_open_start, use_open_end, output_prefix, (T **) &averageSequence, &averageSequenceLength, norm_sequences);

	// Step 4. Save results.
	std::ofstream avg_file(CONCAT2(output_prefix, ".avg.txt").c_str());
	if(!avg_file.is_open()){
		std::cerr << "Cannot open sequence averages file " << output_prefix << ".avg.txt for writing" << std::endl;
		exit(CANNOT_WRITE_DBA_AVG);
	}
	for (size_t i = 0; i < averageSequenceLength; ++i) { avg_file << ((T *) averageSequence)[i] << std::endl; }
	avg_file.close();

	// Cleanup
	for (int i = 0; i < actual_num_series; i++){ 
		cudaFreeHost(series_file_names[i]); CUERR("Freeing CPU memory for a sequence name");
		if(min_segment_length == 0){ // i.e. we still have the original seqs
			cudaFreeHost(sequences[i]); CUERR("Freeing CPU memory for an original sequence");
		}
	}
	cudaFreeHost(series_file_names); CUERR("Freeing CPU memory for the sequence names array");
	cudaFreeHost(sequences); CUERR("Freeing CPU memory for the sequence pointers");
	cudaFreeHost(sequence_lengths); CUERR("Freeing CPU memory for the sequence lengths");
	cudaFreeHost(averageSequence); CUERR("Freeing CPU memory for the DBA result");
}

#endif
