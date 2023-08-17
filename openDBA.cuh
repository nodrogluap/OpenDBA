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
setupAndRun(char *seqprefix_file_name, char **series_file_names, int num_series, char *output_prefix, int read_mode, int use_open_start, int use_open_end, char *min_segment_length_string, int norm_sequences, double cdist, const int prefix_start=0, const int prefix_length=0, bool is_short=false){
	size_t *sequence_lengths = 0;
	T **segmented_sequences = 0;
	size_t *segmented_seq_lengths = 0;
	T **sequences = 0;
	char** sequence_names;
	int actual_num_series = 0; // excludes failed file reading

	// The minimum segment length specified can be either a number to be applied to both clustering and consensus generation like "4", 
	// or two numbers separated by comma like "4,0" which would cluster a segmented sequence but generate consensus on the raw signals from those clusters.
	int min_segment_length;
	int min_segment_length_2 = -1; // -1 is a sentinel for "not defined"
	char *pos = strchr(min_segment_length_string, ',');
	if(pos){ // there was a comma
		min_segment_length_2 = atoi(pos+1);
		*pos = '\0';
	}
	min_segment_length = atoi(min_segment_length_string);

	// Step 0. Read in data.
	if(read_mode == BINARY_READ_MODE){ actual_num_series = readSequenceBinaryFiles<T>(series_file_names, num_series, &sequences, &sequence_names, &sequence_lengths, is_short); }
	// In the following two the sequence names are from inside the file, not the file names themselves
	else if(read_mode == TSV_READ_MODE){ actual_num_series = readSequenceTSVFiles<T>(series_file_names, num_series, &sequences, &sequence_names, &sequence_lengths); }
#if SLOW5_SUPPORTED == 1
	else if(read_mode == SLOW5_READ_MODE){
		actual_num_series = readSequenceSLOW5Files<T>(series_file_names, num_series, &sequences, &sequence_names, &sequence_lengths);
		writeSequences(sequences, sequence_lengths, sequence_names, actual_num_series, CONCAT2(output_prefix, ".seqs.txt").c_str());
	}
#endif	
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
		adaptive_segmentation<T>(sequences, sequence_lengths, actual_num_series, min_segment_length, &segmented_sequences, &segmented_seq_lengths, prefix_start);
		teardownPercentageDisplay();
		int num_seqs_removed = 0;
		for (int i = 0; i < actual_num_series; i++){ 
			// Will we need to revisit the raw sequence?
			if(min_segment_length_2 == -1){cudaFree(sequences[i]); CUERR("Freeing managed memory for a presegmentation sequence");}
			// 1. Sequences of length 1 are problematic as there is no meaningful warp to be performed, and they are almost certain to become the initial medoid.
			// We therefore eliminate them.
			if(segmented_seq_lengths[i-num_seqs_removed] < 2 || prefix_length > 0 && segmented_seq_lengths[i-num_seqs_removed] < prefix_length){
				cudaFree(segmented_sequences[i-num_seqs_removed]); CUERR("Freeing managed memory for a discarded post-segmentation sequence");
				for (int j = i - num_seqs_removed + 1; j < actual_num_series; j++){ 
					segmented_sequences[j-1] = segmented_sequences[j]; // TODO: use memmove() instead?
					segmented_seq_lengths[j-1] = segmented_seq_lengths[j];
				}
				num_seqs_removed++;
			}
		}
		if(num_seqs_removed){
			std::cerr << "Removing " << num_seqs_removed << " segmented sequences that are too short, as they may unduly skew the convergence process. "
				  << "To retain more sequences, consider setting a smaller minimum segment size (currently " 
				  << min_segment_length << ")" << std::endl;
			actual_num_series -= num_seqs_removed;
			if(actual_num_series < 2){
				std::cerr << "At least two sequences must survive segmentation filters to calculate an average, but found " << actual_num_series << ", aborting" << std::endl;
				exit(NOT_ENOUGH_SEQUENCES);
			}
		}
		// 2. Artificially set all the sequence lengths to the requested length for inspection (alignment).
		if(prefix_length > 0){
			for (int i = 0; i < actual_num_series; i++){
				segmented_seq_lengths[i] = prefix_length; 
			}
		}
		writeSequences(segmented_sequences, segmented_seq_lengths, sequence_names, actual_num_series, CONCAT2(output_prefix, ".segmented_seqs.txt").c_str());
		// The user can specify a segmentation size for assigning clusters, then use those cluster memberships to perform centroid convergence with another (or no) segmentation.
		// This could be particularly useful for doing multi-file consensus generation, using a first round of 4 for cluster determination (denoised distances, kind of), then raw cluster consensus generation for each file.
		// The consensus FAST5 files (which will contain fewer "reads" than the originals) could then all be run together for final cluster generation.
		if(min_segment_length_2 != -1){
			std::cerr << "Performing cluster generation with segment size of " << min_segment_length << std::endl;
			performDBA<T>(segmented_sequences, actual_num_series, segmented_seq_lengths, sequence_names, use_open_start, use_open_end, 
				      output_prefix, norm_sequences, cdist, series_file_names, num_series, read_mode, min_segment_length > 1, CLUSTER_ONLY);

			for (int i = 0; i < actual_num_series; i++){
				cudaFree(segmented_sequences[i]); CUERR("Freeing managed memory for a segmented sequence after a clustering-only DBA call");
			}
			cudaFree(segmented_sequences); CUERR("Freeing managed memory for the clustering-only segmentation sequence pointers");
			cudaFree(segmented_seq_lengths); CUERR("Freeing managed memory for the clustering-only sequence lengths");
			// If the clustering step included prefix chopping, and we're doing no segmentation for the consensus generation with FAST5 input, assume we need to reload the raw sequences
			// for consensus generation, as downstream applications like basecaling will want to see that leader/prefix in the data as if the consensus were a raw signal. 
#if SLOW5_SUPPORTED == 1 || HDF5_SUPPORTED == 1
			if(seqprefix_file_name != 0 && min_segment_length_2 == 0 && (
#if HDF5_SUPPORTED == 1
						read_mode == FAST5_READ_MODE 
#endif
#if SLOW5_SUPPORTED == 1 && HDF5_SUPPORTED == 1
						|| 
#endif
#if SLOW5_SUPPORTED == 1
						read_mode == SLOW5_READ_MODE
#endif
						)){
				std::cerr << "Restoring raw signals (no prefix chop) before FAST5/SLOW5 consensus generation without segmentation" << std::endl;
				for (int i = 0; i < actual_num_series; i++){
                                	cudaFree(sequences[i]); CUERR("Freeing managed memory for a prefix-chopped raw sequence after a clustering-only DBA call");
					cudaFreeHost(sequence_names[i]); CUERR("Freeing managed memory for a prefix-chopped raw sequence name after a clustering-only DBA call");
                        	}
				cudaFreeHost(sequence_names); CUERR("Freeing managed memory for the prefix-chopped raw sequence name pointers after a clustering-only DBA call");
				cudaFree(sequences); CUERR("Freeing managed memory for the prefix-chopped raw sequence pointers after a clustering-only DBA call");
				cudaFree(sequence_lengths); CUERR("Freeing managed memory for the prefix-chopped raw sequence lengths after a clustering-only DBA call");
#if SLOW5_SUPPORTED == 1
				if(read_mode == SLOW5_READ_MODE){
                			actual_num_series = readSequenceSLOW5Files<T>(series_file_names, num_series, &sequences, &sequence_names, &sequence_lengths);
        			}
#endif
#if HDF5_SUPPORTED == 1
        			if(read_mode == FAST5_READ_MODE){
                			actual_num_series = readSequenceFAST5Files<T>(series_file_names, num_series, &sequences, &sequence_names, &sequence_lengths);
        			}
#endif
			}
#endif

		}
		else{
			cudaFree(sequences); CUERR("Freeing managed memory for the presegmentation sequence pointers");
			cudaFree(sequence_lengths); CUERR("Freeing managed memory for the presegmentation sequence lengths");
			sequences = segmented_sequences;
			sequence_lengths = segmented_seq_lengths;
		}
	}

	// Step 3. The meat of this meal, running DBA proper!
	if(min_segment_length_2 != -1){
		// Will read the cluster membership info from the segmented seq performDBA call above.
		std::cerr << "Performing consensus generation with segment size of " << min_segment_length_2 << std::endl;
		performDBA<T>(sequences, actual_num_series, sequence_lengths, sequence_names, use_open_start, use_open_end, output_prefix, norm_sequences, cdist, series_file_names, num_series, read_mode, min_segment_length > 1, CONSENSUS_ONLY);
	}
	else{	
		std::cerr << "Performing both clustering and consensus generation with segment size of " << min_segment_length << std::endl;
		performDBA<T>(sequences, actual_num_series, sequence_lengths, sequence_names, use_open_start, use_open_end, output_prefix, norm_sequences, cdist, series_file_names, num_series, read_mode, min_segment_length > 1, CLUSTER_AND_CONSENSUS);
	}

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
