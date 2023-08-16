#ifndef __dba_cpu_utils_included
#define __dba_cpu_utils_included

#include "cuda_utils.hpp"
#include "exit_codes.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <sstream>
#include <string>
#include <cstring>
#include <algorithm>
#include <cerrno>

#if SLOW5_SUPPORTED == 1
#include "submodules/slow5lib/include/slow5/slow5.h"
#endif


#if HDF5_SUPPORTED == 1
extern "C"{
  #include "hdf5.h"
}
#endif

#define CONCAT2(a,b)   (std::string(a)+std::string(b))
#define CONCAT3(a,b,c)   (std::string(a)+std::string(b)+std::string(c))

unsigned int FileRead( std::istream & is, std::vector <char> & buff ) {
    is.read( &buff[0], buff.size() );
    return is.gcount();
}

unsigned int CountLines( const std::vector <char> & buff, int sz ) {
    int newlines = 0;
    const char * p = &buff[0];
    for ( int i = 0; i < sz; i++ ) {
        if ( p[i] == '\n' ) {
            newlines++;
        }
    }
    return newlines;
}

// Function that converts a short to the given template value
// data - the short buffer to be converted
// data_length - the length of the buffer passed in
// returns a new buffer that is of type template with the short data stored in it
template <class T>
short* templateToShort(T* data, size_t data_length){
	short* return_data;
	return_data = (short*)malloc(sizeof(short)*data_length);
	std::transform(data, data + data_length, return_data, [](T s){ return (short)s; });
	return return_data;
}

template <typename T>
int
read_binary_data(const char *binary_file_name, T **output_vals, size_t *num_output_vals, bool is_short=false){

  // See how big the file is, so we can allocate the appropriate buffer
  std::ifstream ifs(binary_file_name, std::ios::binary);
  std::streampos n;
  if(ifs){
    ifs.seekg(0, ifs.end);
    n = ifs.tellg();
    *num_output_vals = n/(is_short?2:sizeof(T)); // special case where shorts will be converted to floats for Z-normalized computation
  }
  else{
    return 1;
  }

  T *out = 0;
  cudaMallocManaged(&out, sizeof(T)*(*num_output_vals)); CUERR("Cannot allocate CPU memory for reading sequence from file");

  ifs.seekg(0, std::ios::beg);
  ifs.read((char *) out, n);
  if(is_short){ // in place expansion from short -> float
	  for(std::streamoff i = (std::streamoff) n/sizeof(short); i >= 0; i--){
		out[i] = (float) *(((short*) out)+i);
	  }
  }

  // Only set the output if all the data was succesfully read in.
  *output_vals = out;
  return 0;
}

template <typename T>
int
read_text_data(const char *text_file_name, T **output_vals, size_t *num_output_vals){

	// Count the number of lines in the file (buffering 1MB on read for speed) so we know how much space to allocate for output_vals
	// std::ios::sync_with_stdio(false); //optimization
	const int SZ = 1024 * 1024;
	std::vector <char> read_buffer( SZ );
	std::ifstream ifs(text_file_name); 
	if(!ifs){
		std::cerr << "Error reading in file " << text_file_name << ", skipping" << std::endl;
		return 1;
	}
	int n = 0;
	while(int sz = FileRead(ifs, read_buffer)) {
		n += CountLines(read_buffer, sz);
	}
	*num_output_vals = n;
	if(n == 0){
		std::cerr << "File " << text_file_name << " is empty or not properly formatted. Skipping." << std::endl;
		ifs.close();
		return 1;
	}

	cudaMallocManaged(output_vals, sizeof(T)*n); CUERR("Cannot allocate CPU memory for reading sequence from text file");
  
	// Read the actual values
	ifs.clear(); // get rid of EOF error state
	ifs.seekg(0, std::ios::beg);
	std::stringstream in;      // Make a stream for the line itself
	std::string line;
	int i = 0;
	while(n--){  // Read line by line
		std::getline(ifs, line); in.str(line);
		in >> (*output_vals)[i++];      // Read the first whitespace-separated token
		in.clear(); // to reuse the stringatream parser
	}

	// Only set the output if all the data was succesfully read in.
	// *output_vals = out;
	ifs.close();
	return 0;
}

#if SLOW5_SUPPORTED == 1

int
scan_slow5_data(const char *slow5_file_name, size_t *num_sequences){

    slow5_file_t *sp = slow5_open(slow5_file_name,"r");
    if(sp==NULL){
       std::cerr << "Could not open SLOW5 file " << slow5_file_name << std::endl;
       return SLOW5_FILE_UNREADABLE;
    }

    int ret=0;
    ret = slow5_idx_load(sp);
    if(ret<0){
       std::cerr << "Could not open SLOW5 file index for " << slow5_file_name << std::endl;
       return SLOW5_FILE_UNREADABLE;
    }

    uint64_t num_reads = 0;
    char **read_ids = slow5_get_rids(sp, &num_reads);
    if(read_ids==NULL){
       std::cerr << "Could not get number of reads from SLOW5 file " << slow5_file_name << std::endl;
	   return SLOW5_FILE_UNREADABLE;	
    }

    slow5_idx_unload(sp);
    slow5_close(sp);

	*num_sequences = num_reads;
	return 0;
}
#endif

#if HDF5_SUPPORTED == 1
int
scan_fast5_data(const char *fast5_file_name, size_t *num_sequences){
	hsize_t num_read_objects;

	/* Open an existing file. */
	hid_t file_id = H5Fopen(fast5_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file_id < 0){
		std::cerr << "Could not open HDF5 file " << fast5_file_name << std::endl;
		return FAST5_FILE_UNREADABLE;
	}
	H5Eset_auto1(NULL, NULL);
	// Old format, one read per file
	hid_t read_group = H5Gopen(file_id, "/Raw/Reads", H5P_DEFAULT);
	if(read_group < 0){ // New formst, multiple reads per file
		read_group = H5Gopen(file_id, "/", H5P_DEFAULT);
	}

	if(read_group < 0 || H5Gget_num_objs(read_group, &num_read_objects)){
		std::cerr << "Could not get read groups from FAST5 (HDF5) file " << fast5_file_name << " so skipping" << std::endl;
		H5Gclose(read_group);
		H5Fclose(file_id);
		return FAST5_FILE_CONTENTS_UNRECOGNIZED;
	}
	hsize_t num_read_objects_rejected = 0;
	char read_subgroup_name[6]; // only care about the first five letters for this check, plus null string terminator
	for(int j = 0; j < num_read_objects; ++j){
		// See if the object looks like a read based on its name
		size_t name_size = H5Gget_objname_by_idx(read_group, j, read_subgroup_name, 6);
		// Should have the form Read_# (old) or read_????? (new)
		if(name_size < 5 || (read_subgroup_name[0] != 'R' && read_subgroup_name[0] != 'r') 
                                 || read_subgroup_name[1] != 'e' 
                                 || read_subgroup_name[2] != 'a' 
                                 || read_subgroup_name[3] != 'd' 
                                 || read_subgroup_name[4] != '_'){ 
			std::cout << "Skipping unexpected HDF5 object " << read_subgroup_name << std::endl;
			num_read_objects_rejected++;
			continue;
		}
	}
	H5Gclose(read_group);
	H5Fclose(file_id);
	*num_sequences = (size_t) (num_read_objects - num_read_objects_rejected);
	return 0;
}
#endif

int
scan_tsv_data(const char *tsv_file_name, size_t *num_sequences){

  	// Count the number of lines in the file (buffering 1MB on read for speed) so we know how much space to allocate for sequence pointers 
  	// std::ios::sync_with_stdio(false); // optimization
  	const int SZ = 1024 * 1024;
  	std::vector <char> read_buffer( SZ );
  	std::ifstream ifs(tsv_file_name, std::ios::binary); // Don't bother translating EOL as we are counting only, so using binary mode (PC + *NIX) 
  	if(!ifs){
    		return 1;
  	}
	int n = 0;
	while(int sz = FileRead(ifs, read_buffer)) {
		n += CountLines(read_buffer, sz);
	}
	*num_sequences = n;
	return 0;
}

#if SLOW5_SUPPORTED == 1

template <typename T>
int
read_slow5_data(const char *slow5_file_name, T **sequences, char **sequence_names, size_t *sequence_lengths){
	int local_seq_count_so_far = 0;

    slow5_file_t *sp = slow5_open(slow5_file_name,"r");
    if(sp==NULL){ // No message, assume scan function called earlier provided these
       return 0;
    }

    slow5_rec_t *rec = NULL;
    int ret=0;

    while((ret = slow5_get_next(&rec,sp)) >= 0){

        ssize_t read_length = rec->len_raw_signal;
		ssize_t name_size = strlen(rec->read_id)+1;

		T *t_seq = 0;
		cudaMallocManaged(&t_seq, sizeof(T)*read_length);  CUERR("Cannot allocate managed memory for SLOW5 signal");
		// Convert the SLOW5 raw shorts to the desired datatype from the template
		for(int j = 0; j < read_length; j++){
			t_seq[j] = (T) rec->raw_signal[j];
		}
		sequences[local_seq_count_so_far] = t_seq;
		sequence_lengths[local_seq_count_so_far] = read_length;
		cudaMallocHost(&sequence_names[local_seq_count_so_far], name_size); CUERR("Cannot allocate CPU memory for reading sequence name from SLOW5 file");
                memcpy(sequence_names[local_seq_count_so_far], rec->read_id, name_size);

		local_seq_count_so_far++;
    }

    if(ret != SLOW5_ERR_EOF){  //check if proper end of file has been reached
		std::cerr << "Error in slow5_get_next: Error code " << ret << std::endl;
	    return SLOW5_FILE_UNREADABLE;
    }

    slow5_rec_free(rec);
    slow5_close(sp);

	return local_seq_count_so_far;
}

#endif

#if HDF5_SUPPORTED == 1
template <typename T>
int
read_fast5_data(const char *fast5_file_name, T **sequences, char **sequence_names, size_t *sequence_lengths){
	int local_seq_count_so_far = 0;

        hid_t file_id = H5Fopen(fast5_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
        if(file_id < 0){ // No message, assume scan function called earlier provided these
                return 0;
        }
	bool old_format = true;
        H5Eset_auto1(NULL, NULL);
        // Old format, one read per file
	hid_t read_group = H5Gopen(file_id, "/Raw/Reads", H5P_DEFAULT);
	if(read_group < 0){ // New formst, multiple reads per file
		read_group = H5Gopen(file_id, "/", H5P_DEFAULT);
		old_format = false;
        }
	hsize_t num_read_objects = 0;
        if(read_group < 0 || H5Gget_num_objs(read_group, &num_read_objects)){
                H5Gclose(read_group);
                H5Fclose(file_id);
                return 0;
        }
       	char *read_subgroup_name = NULL;
        for(int i = 0; i < num_read_objects; ++i){
		ssize_t name_size = H5Gget_objname_by_idx(read_group, i, NULL, 0);
		if(name_size < 5){
			continue; // Too short to be "[Rr]ead_..."
		}
		name_size++; // add space for NULL termination
		errno = 0;
		read_subgroup_name = (char *) (read_subgroup_name == NULL ? std::malloc(name_size) : std::realloc(read_subgroup_name, name_size));
	
		if(errno || (read_subgroup_name == NULL)){
			std::cerr << "Error in malloc/realloc for HDF5 read group name: " << strerror(errno) << std::endl;
			exit(FAST5_CANNOT_MALLOC_READNAME);
		}
		name_size = H5Gget_objname_by_idx(read_group, i, read_subgroup_name, name_size);
		// Should have the form Read_# (old) or read_????? (new)
                if(name_size < 5 || (read_subgroup_name[0] != 'R' && read_subgroup_name[0] != 'r') || read_subgroup_name[1] != 'e' || read_subgroup_name[2] != 'a' || read_subgroup_name[3] != 'd' || read_subgroup_name[4] != '_'){
			std::cerr << "Skipping " << read_subgroup_name << " as it does not follow the naming convention" << std::endl;
                        continue;
                }
		hid_t signal_dataset_id = 0;
		if(old_format){
			signal_dataset_id = H5Dopen(file_id, (CONCAT3("/Raw/Reads/",read_subgroup_name,"/Signal")).c_str(), H5P_DEFAULT);
		}
		else{
			signal_dataset_id = H5Dopen(file_id, (CONCAT3("/",read_subgroup_name,"/Raw/Signal")).c_str(), H5P_DEFAULT);
		}
		if(signal_dataset_id < 0){
			std::cerr << "Skipping " << read_subgroup_name << " Signal, H5DOpen failed" << std::endl;
                        continue;
		}
		hid_t signal_dataspace_id = H5Dget_space(signal_dataset_id);
		if(signal_dataspace_id < 0){
			std::cerr << "Skipping " << read_subgroup_name << " Signal, cannot get the data space" << std::endl;
                        continue;
		}
		const hsize_t read_length = H5Sget_simple_extent_npoints(signal_dataspace_id);
		if(read_length < 1){
			std::cerr << "Skipping " << read_subgroup_name << " with reported Signal length " <<  read_length << std::endl;
			continue;
		}
		hid_t memSpace = H5Screate_simple(1, &read_length, NULL);
		if(memSpace < 0){
			std::cerr << "Failed to create a simple memory space specification in the HDF5 API, please report to the software author(s)." << std::endl;
			exit(FAST5_HDF5_API_ERROR);
		}
		short *sequence_buffer = (short *) std::malloc(sizeof(short)*read_length);
		if(H5Dread(signal_dataset_id, H5T_STD_I16LE, memSpace, signal_dataspace_id, H5P_DEFAULT, sequence_buffer) < 0){
			std::cerr << "Skipping " << read_subgroup_name << ", could not get " << read_length << " Signal from bulk FAST5 (HDF5) file '" << fast5_file_name << "'" << std::endl;
			continue;
		}
		T *t_seq = 0;
		cudaMallocManaged(&t_seq, sizeof(T)*read_length);  CUERR("Cannot allocate managed memory for FAST5 signal");
		// Convert the FAST5 raw shorts to the desired datatype from the template
		for(int j = 0; j < read_length; j++){
			t_seq[j] = (T) sequence_buffer[j];
		}
		free(sequence_buffer);
		sequences[i] = t_seq;
		sequence_lengths[i] = read_length;
		cudaMallocHost(&sequence_names[local_seq_count_so_far], name_size); CUERR("Cannot allocate CPU memory for reading sequence name from FAST5 file");
                memcpy(sequence_names[local_seq_count_so_far], read_subgroup_name, name_size-1); // -1 as not ASCIIZ, we'll put that in manually
		sequence_names[local_seq_count_so_far][name_size-1] = '\0';

		H5Dclose(signal_dataset_id);
		local_seq_count_so_far++;
	}
	if(read_subgroup_name != NULL) free(read_subgroup_name);
	H5Gclose(read_group);
	H5Fclose(file_id);

	return local_seq_count_so_far;
}
#endif

template <typename T>
int
read_tsv_data(const char *tsv_file_name, T **sequences, char **sequence_names, size_t *sequence_lengths){
	int local_seq_count_so_far = 0;
	// One sequence per line, values tab separated.
	
	std::ifstream ifs(tsv_file_name);
	if(!ifs){
                return 0;
        }
	for(std::string line; std::getline(ifs, line); ){
		// Assumption is that the first value is the sequence name (or in the case of the UCR time series archive, the sequence class identifier).
		int numDataColumns = std::count(line.begin(), line.end(), '\t');
		sequence_lengths[local_seq_count_so_far] = numDataColumns;
		T *this_seq;
		cudaMallocManaged(&this_seq, sizeof(T)*numDataColumns); CUERR("Cannot allocate managed memory for reading sequence from TSV file");
		sequences[local_seq_count_so_far] = this_seq;

		std::istringstream iss(line);
		std::string seq_name;
		iss >> seq_name;
		cudaMallocHost(&sequence_names[local_seq_count_so_far], seq_name.length()+1); CUERR("Cannot allocate CPU memory for reading sequence name from TSV file");
		memcpy(sequence_names[local_seq_count_so_far], seq_name.c_str(), seq_name.length()+1);
		int element_count = 0;
    		while(iss.good()){
			iss >> this_seq[element_count++]; // automatically does string -> numeric value conversion
		}
		local_seq_count_so_far++;
	}
	return local_seq_count_so_far;
}

template<typename T>
int readSequenceTSVFiles(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths){

	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? " TSV data file" : " TSV data files");

	// Need two passes: 1st figure out how many sequences there are, then in the 2nd we read the sequences into memory.
	size_t total_seq_count = 0;
	for(int i = 0; i < num_files; ++i){
		size_t seq_count_this_file = 0;
		scan_tsv_data(filenames[i], &seq_count_this_file);
		total_seq_count += seq_count_this_file;
	}
	std::cerr << ", total sequence count " << total_seq_count << std::endl;
        std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
        cudaMallocManaged(sequences, sizeof(T *)*total_seq_count); CUERR("Allocating managed memory for sequence pointers");
        cudaMallocHost(sequence_names, sizeof(char *)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");
        cudaMallocManaged(sequence_lengths, sizeof(size_t)*total_seq_count); CUERR("Allocating managed memory for sequence lengths");

        int dotsPrinted = 0;
        char spinner[4] = { '|', '/', '-', '\\'};
	int actual_count = 0;
        for(int i = 0; i < num_files; ++i){
                int newDotTotal = 100*((float) i/(num_files-1));
                if(newDotTotal > dotsPrinted){
                        for(; dotsPrinted < newDotTotal; dotsPrinted++){
                                std::cerr << "\b.|";
                        }
                }
                else{
                        std::cerr << "\b" << spinner[i%4];
                }

                size_t num_seqs_this_file = read_tsv_data<T>(filenames[i], (*sequences) + actual_count, (*sequence_names) + actual_count, (*sequence_lengths) + actual_count);
		if(num_seqs_this_file < 1){
    			std::cerr << "Error reading in TSV file " << filenames[i] << ", skipping" << std::endl;
		}
		else{
			actual_count += num_seqs_this_file;
		}
        }
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return actual_count;
}

#if SLOW5_SUPPORTED == 1

template<typename T>
int readSequenceSLOW5Files(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths){

	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? " S/BLOW5 file" : " S/BLOW5 files");

	// Need two passes: 1st figure out how many sequences there are, then in the 2nd we read the sequences into memory.
	size_t total_seq_count = 0;
	for(int i = 0; i < num_files; ++i){
		size_t seq_count_this_file = 0;
		scan_slow5_data(filenames[i], &seq_count_this_file);
		total_seq_count += seq_count_this_file;
	}
	std::cerr << ", total sequence count " << total_seq_count << std::endl;
        std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
        cudaMallocManaged(sequences, sizeof(T *)*total_seq_count); CUERR("Allocating managed memory for sequence pointers");
        cudaMallocHost(sequence_names, sizeof(char *)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");
        cudaMallocManaged(sequence_lengths, sizeof(size_t)*total_seq_count); CUERR("Allocating managed memory for sequence lengths");

        int dotsPrinted = 0;
        char spinner[4] = { '|', '/', '-', '\\'};
	int actual_count = 0;
        for(int i = 0; i < num_files; ++i){
                int newDotTotal = 100*((float) i/(num_files-1));
                if(newDotTotal > dotsPrinted){
                        for(; dotsPrinted < newDotTotal; dotsPrinted++){
                                std::cerr << "\b.|";
                        }
                }
                else{
                        std::cerr << "\b" << spinner[i%4];
                }

                size_t num_seqs_this_file = read_slow5_data<T>(filenames[i], (*sequences) + actual_count, (*sequence_names) + actual_count, (*sequence_lengths) + actual_count);
		if(num_seqs_this_file < 1){
    			std::cerr << "Error reading in SLOW5 file " << filenames[i] << ", skipping" << std::endl;
		}
		else{
			actual_count += num_seqs_this_file;
		}
        }
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return actual_count;
}

#endif

#if HDF5_SUPPORTED == 1
template<typename T>
int readSequenceFAST5Files(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths){

        std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? " FAST5 data file" : " FAST5 data files");
	// Need two passes: 1st figure out how many sequences there are, then in the 2nd we read the sequences into memory.
        size_t total_seq_count = 0;
        for(int i = 0; i < num_files; ++i){
                size_t seq_count_this_file = 0;
                scan_fast5_data(filenames[i], &seq_count_this_file);
                total_seq_count += seq_count_this_file;
        }
        std::cerr << ", total sequence count " << total_seq_count << std::endl;
        std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
	cudaMallocManaged(sequences, sizeof(T *)*total_seq_count); CUERR("Allocating managed memory for sequence pointers");
        cudaMallocHost(sequence_names, sizeof(char *)*total_seq_count); CUERR("Allocating CPU memory for sequence names");
        cudaMallocManaged(sequence_lengths, sizeof(size_t)*total_seq_count); CUERR("Allocating managed memory for sequence lengths");

        int dotsPrinted = 0;
        char spinner[4] = { '|', '/', '-', '\\'};
        int actual_count = 0;
        for(int i = 0; i < num_files; ++i){
                int newDotTotal = 100*((float) i/(num_files-1));
                if(newDotTotal > dotsPrinted){
                        for(; dotsPrinted < newDotTotal; dotsPrinted++){
                                std::cerr << "\b.|";
                        }
                }
                else{
                        std::cerr << "\b" << spinner[i%4];
                }

                size_t num_seqs_this_file = read_fast5_data<T>(filenames[i], (*sequences) + actual_count, (*sequence_names) + actual_count, (*sequence_lengths) + actual_count);
                if(num_seqs_this_file < 1){
                        std::cerr << "Error reading in FAST5 file " << filenames[i] << ", skipping" << std::endl;
                }
                else{
                        actual_count += num_seqs_this_file;
                }
        }
        if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
        std::cerr << std::endl;
        return actual_count;
}
#endif

template<typename T>
int readSequenceTextFiles(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths){
        cudaMallocManaged(sequences, sizeof(T *)*num_files); CUERR("Allocating managed memory for sequence pointers from text files");
        cudaMallocHost(sequence_names, sizeof(char *)*num_files); CUERR("Allocating host memory for sequence names from text files");
        cudaMallocManaged(sequence_lengths, sizeof(size_t)*num_files); CUERR("Allocating managed memory for sequence lengths from text files");

        int dotsPrinted = 0;
	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? " text data file" : " text data files") << ", total sequence count " << num_files << std::endl;
        std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
        char spinner[4] = { '|', '/', '-', '\\'};
	int actual_count = 0;
        for(int i = 0; i < num_files; ++i){
                int newDotTotal = 100*((float) i/(num_files-1));
                if(newDotTotal > dotsPrinted){
                        for(; dotsPrinted < newDotTotal; dotsPrinted++){
                                std::cerr << "\b.|";
                        }
                }
                else{
                        std::cerr << "\b" << spinner[i%4];
                }

                if(read_text_data<T>(filenames[i], (*sequences) + actual_count, (*sequence_lengths) + actual_count)){
    			std::cerr << "Error reading in text file " << filenames[i] << ", skipping" << std::endl;
		}
		else{
			actual_count++;
		}

		cudaMallocHost(*sequence_names+i, sizeof(char)*(strlen(filenames[i])+1)); CUERR("Allocating managed memory for a sequence name from text file");
		strcpy((*sequence_names)[i], filenames[i]);
        }
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return actual_count;
}

template<typename T>
int readSequenceBinaryFiles(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths, bool is_short=false){
        cudaMallocManaged(sequences, sizeof(T *)*num_files); CUERR("Allocating CPU memory for sequence pointers from binary files");
	cudaMallocHost(sequence_names, sizeof(char *)*num_files); CUERR("Allocating host memory for sequence names from binary files");
        cudaMallocManaged(sequence_lengths, sizeof(size_t)*num_files); CUERR("Allocating CPU memory for sequence lengths from binary files");

        int dotsPrinted = 0;
	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? " binary data file" : " binary data files") << ", total sequence count " << num_files << std::endl;
        std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
        char spinner[4] = { '|', '/', '-', '\\'};
	int actual_count = 0;
        for(int i = 0; i < num_files; ++i){
                int newDotTotal = 100*((float) i/(num_files-1));
                if(newDotTotal > dotsPrinted){
                        for(; dotsPrinted < newDotTotal; dotsPrinted++){
                                std::cerr << "\b.|";
                        }
                }
                else{
                        std::cerr << "\b" << spinner[i%4];
                }

                if(read_binary_data<T>(filenames[i], (*sequences) + actual_count, (*sequence_lengths) + actual_count, is_short)){
    			std::cerr << "Error reading in binary file " << filenames[i] << ", skipping" << std::endl;
		}
		else{
			actual_count++;
		}
		cudaMallocHost(*sequence_names+i, sizeof(char)*(strlen(filenames[i])+1)); CUERR("Allocating managed memory for a sequence name from text file");
                strcpy((*sequence_names)[i], filenames[i]);
        }
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return actual_count;
}
#endif
