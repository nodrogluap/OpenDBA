#ifndef __io_utils_hpp_included
#define __io_utils_hpp_included

// For definition of DTW moves NIL, RIGHT, UP...
#include "dtw.hpp"
#include "exit_codes.hpp"

// For CONCAT definitions
#include "cpu_utils.hpp"

#if HDF5_SUPPORTED == 1
extern "C"{
  #include "hdf5.h"
}
#endif

#define ARRAYSIZE(a) \
  ((sizeof(a) / sizeof(*(a))) / \
  static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))

// Text progress bar UI element
static char spinner[] = { '|', '/', '-', '\\'};

template <typename T>
__host__
int 
writeSequences(T **cpu_sequences, size_t *seq_lengths, char **seq_names, int num_seqs, const char *filename){
	std::ofstream out(filename);
        if(!out.is_open()){
                std::cerr << "Cannot write to " << filename << std::endl;
                return CANNOT_WRITE_DTW_PATH_MATRIX;
        }
	for(int i = 0; i < num_seqs; i++){
		T * seq = cpu_sequences[i];
		out << seq_names[i] << "\t" << seq[0];
		for(int j = 1; j < seq_lengths[i]; j++){
			out << "\t" << seq[j];
		}
		out << std:: endl;
	}
	out.close();
	return 0;
}

template<typename T>
__host__
int writeDTWPathMatrix(unsigned char *cpu_stepMatrix, const char *step_filename, size_t num_columns, size_t num_rows, size_t pathPitch){
	
	std::ofstream step(step_filename);
	if(!step.is_open()){
		std::cerr << "Cannot write to " << step_filename << std::endl;
		return CANNOT_WRITE_DTW_PATH_MATRIX;
	}	
	
	for(int i = 0; i < num_rows; i++){
		for(int j = 0; j < num_columns; j++){
			unsigned char move = cpu_stepMatrix[pitchedCoord(j,i,pathPitch)];
			step << (move == DIAGONAL ? "D" : (move == RIGHT ? "R" : (move == UP ? "U" : (move == OPEN_RIGHT ?  "O" : (move == NIL || move == NIL_OPEN_RIGHT ? "N" : "?")))));
		}
		step << std::endl;
	}
	step.close();
	
	return 0;
}

template <typename T>
__host__
int writeDTWPath(unsigned char *cpu_pathMatrix, std::ofstream *path, T *gpu_seq, char *cpu_seqname, size_t gpu_seq_len, T *cpu_centroid, size_t cpu_centroid_len, size_t num_columns, size_t num_rows, size_t pathPitch, int flip_seq_order, int column_offset = 0, int *stripe_rows = 0){
	if((*path).tellp() == 0){ // Print the sequence name at the top of the file
		*path << cpu_seqname << std::endl;
	}

	T *cpu_seq;
	// TODO: this is pretty inefficient if running multiple rounds of the same seqs (e.g. during convergence, where we never know if it's the last path that we want to capture), ask CPU seq to be passed in
	cudaMallocHost(&cpu_seq, sizeof(T)*gpu_seq_len); CUERR("Allocating CPU memory for query seq in DTW path printing");
	cudaMemcpy(cpu_seq, gpu_seq, sizeof(T)*gpu_seq_len, cudaMemcpyDeviceToHost); CUERR("Copying incoming GPU query to CPU in DTW path printing");

	// moveI and moveJ are defined device-side in dtw.hpp, but we are host side so we need to replicate
	// NIL sentinel value is for the start of the DTW alignment, the stop condition for backtracking (ergo has no corresponding moveI or moveJ)
	int moveI[] = { -1, -1, 0, -1, 0, 0, 0 };
	int moveJ[] = { -1, -1, -1, 0, -1, -1, -1 };
	int j = num_columns - 1;
	int i = stripe_rows ? *stripe_rows -1 : num_rows - 1;
	unsigned char move = cpu_pathMatrix[pitchedCoord(j,i,pathPitch)];
	while (move != NIL && move != NIL_OPEN_RIGHT && (column_offset == 0 || i >= 0 && j >= 0)) { // special stop condition if partially printing the matrix
        	if(flip_seq_order){
			// Technically NIL and NIL_OPEN_RIGHT should never happen in here, but if they do we know there's a bad bug :-)
                	*path << column_offset+j << "\t" << cpu_seq[j+column_offset] << "\t" << i << "\t" << cpu_centroid[i] << "\t" << (move == DIAGONAL ? "DIAG" : (move == RIGHT ? "RIGHT" : (move == UP ? "UP" : (move == OPEN_RIGHT ? "OPEN_RIGHT" : (move == NIL ? "NIL" : (move == NIL_OPEN_RIGHT ? "NIL_OPEN_RIGHT" : "?")))))) << std::endl;
        	}
        	else{
                	*path << i << "\t" << cpu_seq[i] << "\t" << column_offset+j << "\t" << cpu_centroid[j+column_offset] << "\t" << (move == DIAGONAL ? "DIAG" : (move == RIGHT ? "RIGHT" : (move == UP ? "UP" : (move == OPEN_RIGHT ? "OPEN_RIGHT" : (move == NIL ? "NIL" : (move == NIL_OPEN_RIGHT ? "NIL_OPEN_RIGHT" : "?")))))) << std::endl;
        	}
        	i += moveI[move];
        	j += moveJ[move];
        	move = cpu_pathMatrix[pitchedCoord(j,i,pathPitch)];
	}
	// Print the anchor
	if(column_offset == 0){
		*path << i << "\t" << cpu_seq[i] << "\t" << column_offset+j << "\t" << cpu_centroid[j] << "\t" << (move == NIL ? "NIL" : (move == NIL_OPEN_RIGHT ? "NIL_OPEN_RIGHT" : "?")) << std::endl;
	}
	cudaFreeHost(cpu_seq); CUERR("Freeing CPU memory for query seq in DTW path printing");
	if(stripe_rows){*stripe_rows = i+1;}
	return 0;
}

template <typename T>
__host__
int writePairDistMatrix(char *output_prefix, char **sequence_names, size_t num_sequences, T *dtwPairwiseDistances){
        size_t index_offset = 0;
        std::ofstream mats((std::string(output_prefix)+std::string(".pair_dists.txt")).c_str());
	if(!mats.good()){
		return CANNOT_WRITE_DISTANCE_MATRIX;
	}
        for(size_t seq_index = 0; seq_index < num_sequences-1; seq_index++){
                mats << sequence_names[seq_index];
                for(size_t pad = 0; pad < seq_index; ++pad){
                        mats << "\t";
                }
                mats << "\t0"; //self-distance
                for(size_t paired_seq_index = seq_index + 1; paired_seq_index < num_sequences; ++paired_seq_index){
                        mats << "\t" << dtwPairwiseDistances[index_offset+paired_seq_index-seq_index-1];
                }
                index_offset += num_sequences-seq_index-1;
                mats << std::endl;
        }
        // Last line is pro forma as all pair distances have already been printed
        mats << sequence_names[num_sequences-1];
        for(size_t pad = 0; pad < num_sequences; ++pad){
                mats << "\t";
        }
        mats << "0" << std::endl;

        // Last line is pro forma as all pair distances have already been printed
        mats << sequence_names[num_sequences-1];
        for(size_t pad = 0; pad < num_sequences; ++pad){
                mats << "\t";
        }
        mats << "0" << std::endl;

        mats.close();
	return 0;
}

#if SLOW5_SUPPORTED == 1

// Function to take a SLOW5 file and take a selection of sequences from it. The Raw sequence data of the chosen sequences will be replaced with data passed in from the variable "sequences"
// slow5_file_name - the slow5 file that we will be copying the sequences from
// new_slow5_file - the name of the new slow5 file where the new data will be written
// sequence_names - a list of the sequence names found in the slow5 file that will be copied over
// sequences - the new sequence data that will be written to the new slow5 file
// sequence_lengths - the lengths of the new sequences. These should equal the lengths of the matching sequences in the original slow5 file
// num_sequences - the number of new sequences passed in
// Returns 1 on a fail, 0 on success
__host__
int writeSlow5Output(const char* slow5_file_name, const char* new_slow5_file, char** sequence_names, short** sequences, size_t *sequence_lengths, int num_sequences){
	
    slow5_file_t *sp = slow5_open(slow5_file_name,"r");
    if(sp==NULL){
       std::cerr << "Error opening Slow5 file " << slow5_file_name << " Exiting." << std::endl;
       return 1;
    }
    int ret = slow5_idx_load(sp);
    if(ret<0){
       std::cerr << "Error opening Slow5 index file" << slow5_file_name << " Exiting." << std::endl;
	   return 1;		
	}	
	
    slow5_file_t *sp_new = slow5_open(new_slow5_file, "w");
    if(sp==NULL){
		std::cerr << "Error creating new Slow5 file " << new_slow5_file << " Exiting." << std::endl;
        return 1;
    }
	
	slow5_hdr_t* header_tmp = sp_new->header;
	sp_new->header = sp->header;
	
	
    if(slow5_hdr_write(sp_new) < 0){
		std::cerr << "Error writting header to Slow5 file " << new_slow5_file << " Exiting." << std::endl;
		return 1;	
	}
	
    slow5_rec_t *rec = NULL;

	// Start of reads copy
	for(int i = 0; i < num_sequences; i++){
		
		// Check if sequence exists in original Slow5 file
		ret = slow5_get(sequence_names[i], &rec, sp);
		if(ret < 0){
			std::cerr << "Error. Sequence " << sequence_names[i] << " does not exist in Slow5 file " << slow5_file_name << " Exiting." << std::endl;
			return 1;
		}
			
		// Get the length of the Raw Signal and check if it matches with the length of the new sequence that will be replacing it
		if(rec->len_raw_signal != sequence_lengths[i]){
			std::cerr << "Length of sequence " << sequence_names[i] << " in Slow5 file " << slow5_file_name << " (" << rec->len_raw_signal 
				  << ") does not match length of sequence given (" << sequence_lengths[i] << ") Exiting." << std::endl;
			return 1;
		}	

		for(uint64_t j=0; j < rec->len_raw_signal; j++){
			rec->raw_signal[j]=sequences[i][j];
		}
		
		//write to file
		if (slow5_write(rec, sp_new) < 0){
			std::cerr << "Error writing new sequences to new Slow5 file " << new_slow5_file << " Exiting." << std::endl;
			return 1;
		}	
		
	}

    slow5_rec_free(rec);

    slow5_idx_unload(sp);
    slow5_close(sp);	

	sp_new->header=header_tmp;
	slow5_close(sp_new);


	
	
	return 0;
}
#endif

#if HDF5_SUPPORTED == 1

// Function to take a multi Fast5 file and take a selection of sequences from it. The Raw sequence data of the chosen sequences will be replaced with data passed in from the variable "sequences"
// fast5_file_name - the fast5 file that we will be copying the sequences from
// new_fast5_file - the name of the new fast5 file where the new data will be written
// sequence_names - a list of the sequence names found in the fast5 file that will be copied over
// sequences - the new sequence data that will be written to the new fast5 file
// sequence_lengths - the lengths of the new sequences. These should equal the lengths of the matching sequences in the original fast5 file
// num_sequences - the number of new sequences passed in
// Returns 1 on a fail, 0 on success
__host__
int writeFast5Output(const char* fast5_file_name, const char* new_fast5_file, char** sequence_names, short** sequences, size_t *sequence_lengths, int num_sequences){
	
	// HDF5 variables needed
	hid_t org_file_id, new_file_id, org_read_group, new_read_group, org_attr, new_attr, memtype, space, org_signal_dataset_id, signal_dataspace_id, new_group, new_dataset_prop_list, new_dataset;
	hsize_t org_attr_size;
	
	// Initial copy of metadata
	// Not sure if you can actually 'copy' this info, so for right now we're creating new data in the new file. Might want to see if this can just be copied in the future
	
	// Open Fast5 file we want to copy data from
        if((org_file_id = H5Fopen(fast5_file_name, H5F_ACC_RDONLY, H5P_DEFAULT)) < 0){ 
		std::cerr << "Error opening Fast5 file " << fast5_file_name << " Exiting." << std::endl;
                return 1;
        }
	
	// Create the new Fast5 file we want to write coppied data to
	if((new_file_id = H5Fcreate(new_fast5_file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)) < 0){ 
		std::cerr << "Error creating new Fast5 file " << new_fast5_file << " Exiting." << std::endl;
                return 1;
        }
        H5Eset_auto1(NULL, NULL);
	
	// Open the base read group inside the original file
	if((org_read_group = H5Gopen(org_file_id, "/", H5P_DEFAULT)) < 0){ 
		std::cerr << "Error opening Group '/' from " << fast5_file_name << " Exiting." << std::endl;
                return 1;
        }

	// Open the base read group in the new file
        if((new_read_group = H5Gopen(new_file_id, "/", H5P_DEFAULT)) < 0){
                std::cerr << "Error opening Group '/' from " << new_fast5_file << " Exiting." << std::endl;
                return 1;
        }

	// Open the file_type attribute inside of the / read group in the original file
	// To be closed after use
	if((org_attr = H5Aopen(org_read_group, "file_type", H5P_DEFAULT)) >= 0){
	
		// Get the size of the data in file_type in the original file
		if((org_attr_size = H5Aget_storage_size(org_attr)) == 0){ 
			std::cerr << "Error getting Attribute 'file_type' size from " << fast5_file_name << " Exiting." << std::endl;
                	return 1;
        	}
	
		// Get the type of the data in file_type in the original file
		hid_t org_atype = H5Aget_type(org_attr);
		char* data_buffer = (char*) std::malloc(org_attr_size);
	
		// Read in the data from the attribute file_type in the original file
		if(H5Aread(org_attr, org_atype, (void*)data_buffer)){
			std::cerr << "Error reading attribute 'file_type' from Fast5 file " << fast5_file_name << " Exiting." << std::endl;
			return 1;
		}
	
	
		// Set the memory type of the new attribute to be a c string
		memtype = H5Tcopy(H5T_C_S1);
		if(H5Tset_size(memtype, org_attr_size)){
			std::cerr << "Error assigning size for file_type in " << new_fast5_file << " Exiting." << std::endl;
                	return 1;
		}
	
		// Create a new string scalar dataspace for the data
		if((space = H5Screate(H5S_SCALAR)) < 0){
			std::cerr << "Failed to create a scalar memory space specification in the HDF5 API, please report to the software author(s)." << std::endl;
			exit(FAST5_HDF5_API_ERROR);
        	}
	
		// Create the new attribute with the above memtype and space
		// To be closed after use
		if((new_attr = H5Acreate(new_read_group, "file_type", memtype, space, H5P_DEFAULT, H5P_DEFAULT)) < 0){ 
			std::cerr << "Error creating attribute 'file_type' for " << new_fast5_file << " Exiting." << std::endl;
                	return 1;
        	}
	
		// Write the new attribute to the new file
		if(H5Awrite(new_attr, memtype, (void*)data_buffer)){
			std::cerr << "Error writting attribute 'file_type' to Fast5 file " << new_fast5_file << " Exiting." << std::endl;
			return 1;
		}
	
		H5Aclose(org_attr);
		H5Aclose(new_attr);
	
		// Open 'file_version' attribute in original file
		if((org_attr = H5Aopen(org_read_group, "file_version", H5P_DEFAULT)) < 0){ 
			std::cerr << "Error opening Attribute 'file_version' from " << fast5_file_name << " Exiting." << std::endl;
                	return 1;
        	}
	
		// Get size of 'file_version' attribute
		if((org_attr_size = H5Aget_storage_size(org_attr)) == 0){ 
			std::cerr << "Error getting Attribute 'file_version' size from " << fast5_file_name << " Exiting." << std::endl;
                	return 1;
        	}
	
		org_atype = H5Aget_type(org_attr);
		data_buffer = (char*) std::realloc(data_buffer, org_attr_size);
	
		// Read in the data from the attribute file_version in the original file
		if(H5Aread(org_attr, org_atype, (void*)data_buffer)){
			std::cerr << "Error reading attribute 'file_version' from Fast5 file " << fast5_file_name << " Exiting." << std::endl;
			return 1;
		}
	
		if(H5Tset_size (memtype, org_attr_size)){
			std::cerr << "Error assigning size for file_version in " << new_fast5_file << " Exiting." << std::endl;
                	return 1;
		}
	
		// Create the new attribute with the above memtype and space
		if((new_attr = H5Acreate(new_read_group, "file_version", memtype, space, H5P_DEFAULT, H5P_DEFAULT)) < 0){ 
			std::cerr << "Error creating attribute 'file_version' for " << new_fast5_file << " Exiting." << std::endl;
                	return 1;
        	}
	
		// Write the new attribute to the new file
		if(H5Awrite(new_attr, memtype, (void*)data_buffer)){
			std::cerr << "Error writting attribute 'file_version' to Fast5 file " << new_fast5_file << " Exiting." << std::endl;
			return 1;
		}
	

		// Close what's no longer needed
		H5Tclose(memtype);
		H5Sclose(space);
	
		H5Aclose(org_attr);
		H5Aclose(new_attr);
	
		free(data_buffer);
	}

	H5Gclose(org_read_group);
	H5Gclose(new_read_group);
	
		
	// End of metadata copy
	// Start of reads copy
	
	for(int i = 0; i < num_sequences; i++){
		
		// Check if sequence exists in original Fast5 file
		if(H5Oexists_by_name(org_file_id, CONCAT2("/", sequence_names[i]).c_str(), H5P_DEFAULT) == 0){
			std::cerr << "Error. Sequence " << sequence_names[i] << " does not exist in Fast5 file " << fast5_file_name << " Exiting." << std::endl;
			return 1;
		}
		
		// Get Dataset for the Raw Signal of the sequence
		if((org_signal_dataset_id = H5Dopen(org_file_id, (CONCAT3("/", sequence_names[i], "/Raw/Signal")).c_str(), H5P_DEFAULT)) < 0){
			std::cerr << "Unable to open " << sequence_names[i] << " Signal in " << fast5_file_name << " Exiting." << std::endl;
			return 1;
		}
		
		// Get the Dataspace for the Raw Signal
		if((signal_dataspace_id = H5Dget_space(org_signal_dataset_id)) < 0){
			std::cerr << "Unable to get dataspace for " << sequence_names[i] << " Signal in " << fast5_file_name << " Exiting." << std::endl;
			return 1;
		}
		
		// Get the length of the Raw Signal and check if it matches with the length of the new sequence that will be replacing it
		const hsize_t read_length = H5Sget_simple_extent_npoints(signal_dataspace_id);
		if(read_length != sequence_lengths[i]){
			std::cerr << "Length of sequence " << sequence_names[i] << " in Fast5 file " << fast5_file_name << " (" << read_length 
				  << ") does not match length of sequence given (" << sequence_lengths[i] << ") Exiting." << std::endl;
			return 1;
		}
		
		// Copy over the data of the sequence in the original Fast5 file to the new Fast5 file
		if(H5Ocopy(org_file_id, CONCAT2("/", sequence_names[i]).c_str(), new_file_id, CONCAT2("/", sequence_names[i]).c_str(), H5P_DEFAULT, H5P_DEFAULT)){
			std::cerr << "Error copying Attribute 'file_type' from " << fast5_file_name << " to " << new_fast5_file << " Exiting." << std::endl;
			return 1;
		}
		
		// Create a space that will be the size of the sequence length and have a max size of H5S_UNLIMITED
		hsize_t max_dims = H5S_UNLIMITED;
		if((space = H5Screate_simple(1, &read_length, &max_dims)) < 0){
			std::cerr << "Failed to create a simple memory space specification in the HDF5 API, please report to the software author(s)." << std::endl;
			exit(FAST5_HDF5_API_ERROR);
		}
		
		// Delete the link to the original signal data in the new file
		if(H5Ldelete(new_file_id, (CONCAT3("/", sequence_names[i], "/Raw/Signal")).c_str(), H5P_DEFAULT)){
			std::cerr << "Unable to delete " << (CONCAT3("/", sequence_names[i], "/Raw/Signal")).c_str() << " from " << new_fast5_file << " Exiting." << std::endl;
			return 1;
		}
		
		// Open the group where the new signal data will be written
		if((new_group = H5Gopen(new_file_id, (CONCAT3("/", sequence_names[i], "/Raw")).c_str(), H5P_DEFAULT)) < 0){
			std::cerr << "Unable to open " << (CONCAT3("/", sequence_names[i], "/Raw")).c_str() << " group in " << new_fast5_file << " Exiting." << std::endl;
			return 1;
		}
		
		// Create a property list for the dataset
		if((new_dataset_prop_list = H5Pcreate(H5P_DATASET_CREATE)) < 0){
			std::cerr << "Unable to create property list for " << new_fast5_file << " Exiting." << std::endl;
			return 1;
		}
		
		// Set chunk size for the property list since the max size for the space is UNLIMITED
		if(H5Pset_chunk(new_dataset_prop_list, 1, &read_length)){
			std::cerr << "Unable to set chunk for property list in " << new_fast5_file << " Exiting." << std::endl;
			return 1;
		}
		
		// Create the new dataset with the above parameters that we've set
		if((new_dataset = H5Dcreate2 (new_group, "Signal", H5T_STD_I16LE, space, H5P_DEFAULT, new_dataset_prop_list, H5P_DEFAULT)) < 0){
			std::cerr << "Unable to create dataset " << (CONCAT3("/", sequence_names[i], "/Raw/Signal")).c_str() << " in " << new_fast5_file << " Exiting." << std::endl;
			return 1;
		}
		
		// Write the new sequences to the newly created dataset
		if(H5Dwrite(new_dataset, H5T_STD_I16LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, sequences[i])){
			std::cerr << "Error writing new sequences to new Fast5 file " << new_fast5_file << " Exiting." << std::endl;
			return 1;
		}
		
		// Close what was opened in the loop
		H5Gclose(new_group);
		H5Pclose(new_dataset_prop_list);
		H5Dclose(new_dataset);
		H5Dclose(org_signal_dataset_id);
		H5Sclose(space);
	}
	
	
	// Close everything else
	H5Fclose(org_file_id);
	H5Fclose(new_file_id);
	
	return 0;
}

#endif

__host__
void setupPercentageDisplay(std::string title){
	std::cerr << title << std::endl;
        std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
}

__host__
void teardownPercentageDisplay(){
        std::cerr << std::endl;
}

__host__
int updatePercentageComplete(int current_item, int total_items, int alreadyDisplaying){
	int newDisplayTotal = 100*((float) current_item/total_items);
	if(newDisplayTotal > alreadyDisplaying){
		for(; alreadyDisplaying < newDisplayTotal; alreadyDisplaying++){
			std::cerr << "\b.|";
		}
	}
	else{
		std::cerr << "\b" << spinner[current_item%ARRAYSIZE(spinner)];
	}
	return newDisplayTotal;
}
#endif
