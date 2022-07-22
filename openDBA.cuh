#ifndef OPENDBA_H
#define OPENDBA_H

#include <string.h>
#include <iostream>
#include <fstream>
#include "cpu_utils.hpp"
#include "dba.hpp"
#include "segmentation.hpp"
#include "io_utils.hpp"
#include "read_mode_codes.h"

template<typename T>
void
setupAndRun(char *seqprefix_file_name, char **series_file_names, int num_series, char *output_prefix, int read_mode, int use_open_start, int use_open_end, int min_segment_length, int norm_sequences, double cdist, bool is_short=false){
	size_t *sequence_lengths = 0;
	T **segmented_sequences = 0;
	size_t *segmented_seq_lengths = 0;
	T **sequences = 0;
	char** sequence_names;
	int actual_num_series = 0; // excludes failed file reading

	// Step 0. Read in data.
	if(read_mode == BINARY_READ_MODE){ actual_num_series = readSequenceBinaryFiles<T>(series_file_names, num_series, &sequences, &sequence_names, &sequence_lengths, is_short); }
	// In the following two the sequence names are from inside the file, not the file names themselves
	else if(read_mode == TSV_READ_MODE){ actual_num_series = readSequenceTSVFiles<T>(series_file_names, num_series, &sequences, &sequence_names, &sequence_lengths); }
#if HDF5_SUPPORTED == 1
	else if(read_mode == FAST5_READ_MODE){ 
		actual_num_series = readSequenceFAST5Files<T>(series_file_names, num_series, &sequences, &sequence_names, &sequence_lengths); 
		writeSequences(sequences, sequence_lengths, sequence_names, actual_num_series, CONCAT2(output_prefix, ".seqs.txt").c_str());
	}
#endif
	else{ actual_num_series = readSequenceTextFiles<T>(series_file_names, num_series, &sequences, &sequence_names, &sequence_lengths); }

	// Sanity check
	if(actual_num_series < 2){
		std::cerr << "At least two sequences must be provided to calculate an average, but found " << actual_num_series << ", aborting" << std::endl;
		exit(NOT_ENOUGH_SEQUENCES);
	}

	// Shorten sequence names to everything before the first "." in the file name
	for (int i = 0; i < actual_num_series; i++){ char *z = strchr(sequence_names[i], '.'); if(z) *z = '\0';}

	// Step 1. If a leading sequence was specified, chop it off all the inputs.
	if(seqprefix_file_name != 0){
		T **seqprefix = 0;
		size_t *seqprefix_length = 0;
		char** seqprefix_name;
		if(read_mode == BINARY_READ_MODE){
			readSequenceBinaryFiles<T>(&seqprefix_file_name, 1, &seqprefix, &seqprefix_name, &seqprefix_length);
		}
		else{
			readSequenceTextFiles<T>(&seqprefix_file_name, 1, &seqprefix, &seqprefix_name, &seqprefix_length);
		}
		if(*seqprefix_length == 0){
			std::cerr << "Cannot read prefix " << (read_mode == BINARY_READ_MODE ? "binary" : "text") << 
				" data from " << seqprefix_file_name << ", aborting" << std::endl;
			exit(CANNOT_READ_SEQUENCE_PREFIX_FILE);
		}
		setupPercentageDisplay("Opt-in Step: Chopping sequence prefixes");
		chopPrefixFromSequences<T>(*seqprefix, *seqprefix_length, sequences, &actual_num_series, sequence_lengths, sequence_names, output_prefix, norm_sequences);
		teardownPercentageDisplay();
		cudaFree(*seqprefix); CUERR("Freeing managed memory for the prefix sequence");
		cudaFree(seqprefix); CUERR("Freeing managed memory for the prefix sequencers pointer");
		cudaFree(seqprefix_length); CUERR("Freeing managed memory for the prefix sequence length");
	}
	// Step 2. If a minimum segment length was provided, segment the input sequences into unimodal pieces. 
	if(min_segment_length > 0){
		setupPercentageDisplay("Opt-in Step: Segmenting with minimum acceptable segment size of " + std::to_string(min_segment_length));
		adaptive_segmentation<T>(sequences, sequence_lengths, actual_num_series, min_segment_length, &segmented_sequences, &segmented_seq_lengths);
		teardownPercentageDisplay();
		for (int i = 0; i < actual_num_series; i++){ 
			cudaFree(sequences[i]); CUERR("Freeing managed memory for a presegmentation sequence");
			// Sequences of length 1 are problematic as there is no meaningful warp to be performed, and they are almost certain to become the initial medoid.
			// We therefore eliminate them.
			if(segmented_seq_lengths[i] < 2){
				std::cerr << "Removing segmented sequence '" << sequence_names[i] << "' of length " << segmented_seq_lengths[i]
					  << " as it may unduly skew the convergence process. To retain this sequence, consider setting a smaller minimum segment size (currently " 
					  << min_segment_length << ")" << std::endl;
				cudaFree(segmented_sequences[i]); CUERR("Freeing managed memory for a discarded post-segmentation sequence");
				for (int j = i + 1; j < actual_num_series; j++){ 
					segmented_sequences[j-1] = segmented_sequences[j]; // TODO: use memmove() instead?
					segmented_seq_lengths[j-1] = segmented_seq_lengths[j];
				}
				actual_num_series--;
			}
		}
		writeSequences(segmented_sequences, segmented_seq_lengths, sequence_names, actual_num_series, CONCAT2(output_prefix, ".segmented_seqs.txt").c_str());
		cudaFree(sequences); CUERR("Freeing managed memory for the presegmentation sequence pointers");
		cudaFree(sequence_lengths); CUERR("Freeing managed memory for the presegmentation sequence lengths");
		sequences = segmented_sequences;
		sequence_lengths = segmented_seq_lengths;
	}

	// Step 3. The meat of this meal!
	performDBA<T>(sequences, actual_num_series, sequence_lengths, sequence_names, use_open_start, use_open_end, output_prefix, norm_sequences, cdist, series_file_names, num_series, read_mode);

	// Cleanup
	for (int i = 0; i < actual_num_series; i++){ 
		cudaFreeHost(sequence_names[i]); CUERR("Freeing CPU memory for a sequence name");
		if(min_segment_length == 0){ // i.e. we still have the original seqs
			cudaFree(sequences[i]); CUERR("Freeing managed memory for an original sequence");
		}
	}
	cudaFreeHost(sequence_names); CUERR("Freeing CPU memory for the sequence names array");
	cudaFree(sequences); CUERR("Freeing managed memory for the sequence pointers");
	cudaFree(sequence_lengths); CUERR("Freeing managed memory for the sequence lengths");
}

#endif
