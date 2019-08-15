#ifndef __dba_cpu_utils_included
#define __dba_cpu_utils_included

#include "cuda_utils.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <sstream>
#include <string>
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
    std::cerr << "Error reading in file " << binary_file_name << ", skipping" << std::endl;
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

  // Count the number of lines in the file (buffering 1MB on read for speed) so we now how much space to allocate for output_vals
  std::ios::sync_with_stdio(false); //optimization
  const int SZ = 1024 * 1024;
  std::vector <char> read_buffer( SZ );
  std::ifstream ifs(text_file_name, std::ios::binary); // Don't bother translating EOL as we are counting only, so using binary mode (PC + *NIX) 
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
  cudaMallocHost(&out, sizeof(T)*n); CUERR("Cannot allocate CPU memory for reading sequence from file");
  
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

template<typename T>
void readSequenceTextFiles(char **filenames, int num_files, T ***sequences, size_t **sequence_lengths){
        cudaMallocHost(sequences, sizeof(T **)*num_files); CUERR("Allocating CPU memory for sequence pointers");
        cudaMallocHost(sequence_lengths, sizeof(size_t)*num_files); CUERR("Allocating CPU memory for sequence lengths");

        int dotsPrinted = 0;
	std::cerr << "Step 1 of 3: Loading " << num_files << " data files" << std::endl;
        std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
        char spinner[4] = { '|', '/', '-', '\\'};
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

                if(read_text_data<T>(filenames[i], (*sequences) + i, (*sequence_lengths) + i)){
			i--;
		}
        }
	std::cerr << std::endl;
}

template<typename T>
void readSequenceBinaryFiles(char **filenames, int num_files, T ***sequences, size_t **sequence_lengths){
        cudaMallocHost(sequences, sizeof(T **)*num_files); CUERR("Allocating CPU memory for sequence pointers");
        cudaMallocHost(sequence_lengths, sizeof(size_t)*num_files); CUERR("Allocating CPU memory for sequence lengths");

        int dotsPrinted = 0;
        std::cerr << "Step 1 of 3: Loading " << num_files << " data files" << std::endl;
        std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
        char spinner[4] = { '|', '/', '-', '\\'};
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

                if(read_binary_data<T>(filenames[i], (*sequences) + i, (*sequence_lengths) + i)){
			i--;
		}
        }
	std::cerr << std::endl;
}
#endif
