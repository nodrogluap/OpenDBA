#ifndef __dba_cpu_utils_included
#define __dba_cpu_utils_included

#include "cuda_utils.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <sstream>
#include <string>
#include <cstring>
#include <algorithm>
//using namespace std;

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

	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? " TSV data file" : "TSV data files");

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
                int newDotTotal = 100*((float) actual_count/(num_files-1));
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
    			std::cerr << "Error reading in file " << filenames[i] << ", skipping" << std::endl;
		}
		else{
			actual_count += num_seqs_this_file;
		}
        }
	while(dotsPrinted++ < 100){std::cerr << ".";}
	std::cerr << std::endl;
	return actual_count;
}

template<typename T>
int readSequenceTextFiles(char **filenames, int num_files, T ***sequences, size_t **sequence_lengths){
        cudaMallocHost(sequences, sizeof(T *)*num_files); CUERR("Allocating CPU memory for sequence pointers");
        cudaMallocHost(sequence_lengths, sizeof(size_t)*num_files); CUERR("Allocating CPU memory for sequence lengths");

        int dotsPrinted = 0;
	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? "text data file" : "text data files") << ", total sequence count " << num_files << std::endl;
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
	while(dotsPrinted++ < 100){std::cerr << ".";}
	std::cerr << std::endl;
	return actual_count;
}

template<typename T>
int readSequenceBinaryFiles(char **filenames, int num_files, T ***sequences, size_t **sequence_lengths){
        cudaMallocHost(sequences, sizeof(T *)*num_files); CUERR("Allocating CPU memory for sequence pointers");
        cudaMallocHost(sequence_lengths, sizeof(size_t)*num_files); CUERR("Allocating CPU memory for sequence lengths");

        int dotsPrinted = 0;
	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? "binary data file" : "binary data files") << ", total sequence count " << num_files << std::endl;
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
	while(dotsPrinted++ < 100){std::cerr << ".";}
	return actual_count;
}
#endif
