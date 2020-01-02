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

template <typename T>
int
read_binary_data(const char *binary_file_name, T **output_vals, size_t *num_output_vals){

  // See how big the file is, so we can allocate the appropriate buffer
  std::ifstream ifs(binary_file_name, std::ios::binary);
  std::streampos n;
  if(ifs){
    ifs.seekg(0, ifs.end);
    n = ifs.tellg();
    *num_output_vals = n/sizeof(T);
  }
  else{
    return 1;
  }

  T *out = 0;
  cudaMallocHost(&out, sizeof(T)*n); CUERR("Cannot allocate CPU memory for reading sequence from file");

  ifs.seekg(0, std::ios::beg);
  ifs.read((char *) out, n);

  // Only set the output if all the data was succesfully read in.
  *output_vals = out;
  return 0;
}

template <typename T>
int
read_text_data(const char *text_file_name, T **output_vals, size_t *num_output_vals){

  // Count the number of lines in the file (buffering 1MB on read for speed) so we know how much space to allocate for output_vals
  std::ios::sync_with_stdio(false); //optimization
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
    return 1;
  }

  T *out = 0;
  cudaMallocHost(&out, sizeof(T)*n); CUERR("Cannot allocate CPU memory for reading sequence from text file");
  
  // Read the actual values
  ifs.clear(); // get rid of EOF error state
  ifs.seekg(0, std::ios::beg);
  std::stringstream in;      // Make a stream for the line itself
  std::string line;
  int i = 0;
  while(n--){  // Read line by line
    std::getline(ifs, line); in.str(line);
    in >> out[i++];      // Read the first whitespace-separated token
    in.clear(); // to reuse the stringatream parser
  }

  // Only set the output if all the data was succesfully read in.
  *output_vals = out;
  return 0;
}

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
		H5Fclose(file_id);
		H5Gclose(read_group);
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
	*num_sequences = (size_t) (num_read_objects - num_read_objects_rejected);
	return 0;
}
#endif

int
scan_tsv_data(const char *text_file_name, size_t *num_sequences){

  	// Count the number of lines in the file (buffering 1MB on read for speed) so we know how much space to allocate for sequence pointers 
  	std::ios::sync_with_stdio(false); // optimization
  	const int SZ = 1024 * 1024;
  	std::vector <char> read_buffer( SZ );
  	std::ifstream ifs(text_file_name, std::ios::binary); // Don't bother translating EOL as we are counting only, so using binary mode (PC + *NIX) 
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
                H5Fclose(file_id);
                H5Gclose(read_group);
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
		cudaMallocHost(&t_seq, sizeof(T)*read_length);  CUERR("Cannot allocate CPU memory for FAST5 signal");
		// Convert the FAST5 raw shorts to the desired datatype from the template
		for(int j = 0; j < read_length; j++){
			t_seq[j] = (T) sequence_buffer[j];
		}
		free(sequence_buffer);
		sequences[i] = t_seq;
		sequence_lengths[i] = read_length;
		cudaMallocHost(&sequence_names[local_seq_count_so_far], name_size); CUERR("Cannot allocate CPU memory for reading sequence name from FAST5 file");
                memcpy(sequence_names[local_seq_count_so_far], read_subgroup_name, name_size);

		H5Dclose(signal_dataset_id);
		local_seq_count_so_far++;
	}
	if(read_subgroup_name != NULL) free(read_subgroup_name);
	return local_seq_count_so_far;
}
#endif

template <typename T>
int
read_tsv_data(const char *text_file_name, T **sequences, char **sequence_names, size_t *sequence_lengths){
	int local_seq_count_so_far = 0;
	// One sequence per line, values tab separated.
	
	std::ifstream ifs(text_file_name);
	if(!ifs){
                return 0;
        }
	for(std::string line; std::getline(ifs, line); ){
		// Assumption is that the first value is the sequence name (or in the case of the UCR time series archive, the sequence class identifier).
		int numDataColumns = std::count(line.begin(), line.end(), '\t');
		sequence_lengths[local_seq_count_so_far] = numDataColumns;
		T *this_seq;
		cudaMallocHost(&this_seq, sizeof(T)*numDataColumns); CUERR("Cannot allocate CPU memory for reading sequence from TSV file");
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
        cudaMallocHost(sequences, sizeof(T *)*total_seq_count); CUERR("Allocating CPU memory for sequence pointers");
        cudaMallocHost(sequence_names, sizeof(char *)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");
        cudaMallocHost(sequence_lengths, sizeof(size_t)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");

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
	cudaMallocHost(sequences, sizeof(T *)*total_seq_count); CUERR("Allocating CPU memory for sequence pointers");
        cudaMallocHost(sequence_names, sizeof(char *)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");
        cudaMallocHost(sequence_lengths, sizeof(size_t)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");

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
int readSequenceTextFiles(char **filenames, int num_files, T ***sequences, size_t **sequence_lengths){
        cudaMallocHost(sequences, sizeof(T *)*num_files); CUERR("Allocating CPU memory for sequence pointers");
        cudaMallocHost(sequence_lengths, sizeof(size_t)*num_files); CUERR("Allocating CPU memory for sequence lengths");

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
    			std::cerr << "Error reading in file " << filenames[i] << ", skipping" << std::endl;
		}
		else{
			actual_count++;
		}
        }
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return actual_count;
}

template<typename T>
int readSequenceBinaryFiles(char **filenames, int num_files, T ***sequences, size_t **sequence_lengths){
        cudaMallocHost(sequences, sizeof(T *)*num_files); CUERR("Allocating CPU memory for sequence pointers");
        cudaMallocHost(sequence_lengths, sizeof(size_t)*num_files); CUERR("Allocating CPU memory for sequence lengths");

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

                if(read_binary_data<T>(filenames[i], (*sequences) + actual_count, (*sequence_lengths) + actual_count)){
    			std::cerr << "Error reading in file " << filenames[i] << ", skipping" << std::endl;
		}
		else{
			actual_count++;
		}
        }
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return actual_count;
}
#endif
