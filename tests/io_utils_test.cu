#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <stdlib.h>     /* srand, rand */
#include <string>

#include <iostream>
#include <fstream>

#if defined(_WIN32)
	#include <direct.h>
	#include <conio.h>
	#include <windows.h>
	#include <bitset>
	extern "C"{
		#include "getopt.h"
	}
	#define GetCurrentDir _getcwd
	#define ONWINDOWS 1
#else
	#include <unistd.h>
	#define GetCurrentDir getcwd
	#define ONWINDOWS 0
#endif

#include "../io_utils.hpp"
#include "../cpu_utils.hpp"

#include "test_utils.cuh"

char* cur_dir_char = (char*) malloc(FILENAME_MAX);
char* tmp = GetCurrentDir( cur_dir_char, FILENAME_MAX );
std::string current_working_dir(cur_dir_char);

// Function that converts a string to char pointer
// tmp_string - the string we want to convert_dna_to_shorts
// returns converted char pointer
char* stringToChar(std::string tmp_string){
	char* cstr = (char*) malloc(tmp_string.size() + 1);
	strcpy(cstr, tmp_string.c_str());
	return cstr;
}

TEST_CASE( " Write Fast5 Output " ){
	
	std::string series_filename_seq = current_working_dir + "/good_files/fast5/FAN41461_pass_496845aa_0.fast5";
	std::string fast5_output = current_working_dir + "/test_output.fast5";
	
	std::string good_file = current_working_dir + "/good_files/tsv/openDBA_test_edit_fast5.tsv";	
	std::string wrong_size_file = current_working_dir + "/wrong_files/tsv/openDBA_test_size_fast5.tsv";
	std::string wrong_name_file = current_working_dir + "/wrong_files/tsv/openDBA_test_name_fast5.tsv";
	
	char** filenames = (char**)malloc(sizeof(char*)*4);
	filenames[0] = stringToChar(good_file);
	filenames[1] = stringToChar(wrong_size_file);
	filenames[2] = stringToChar(wrong_name_file);
	
	
	SECTION("Good File Data"){
		
		std::cerr << "------TEST 1: successful raw signal replacement with same length data ------" << std::endl;
		
		short **sequences;
		char ** sequence_names;
		size_t *sequence_lengths;
	
		cudaMallocHost(&sequences, sizeof(short *)*3); CUERR("Allocating CPU memory for sequence pointers");
		cudaMallocHost(&sequence_names, sizeof(char *)*3); CUERR("Allocating CPU memory for sequence lengths");
		cudaMallocHost(&sequence_lengths, sizeof(size_t)*3); CUERR("Allocating CPU memory for sequence lengths");
		
		int num_sequences = read_tsv_data<short>(filenames[0], sequences, sequence_names, sequence_lengths);
		
		int result = writeFast5Output(stringToChar(series_filename_seq), stringToChar(fast5_output), sequence_names, sequences, sequence_lengths, num_sequences);
		
		short **result_sequences;
		char ** result_sequence_names;
		size_t *result_sequence_lengths;
		
		cudaMallocHost(&result_sequences, sizeof(short *)*3); CUERR("Allocating CPU memory for sequence pointers");
		cudaMallocHost(&result_sequence_names, sizeof(char *)*3); CUERR("Allocating CPU memory for sequence lengths");
		cudaMallocHost(&result_sequence_lengths, sizeof(size_t)*3); CUERR("Allocating CPU memory for sequence lengths");
		
		int num_result_sequences = read_fast5_data(stringToChar(fast5_output), result_sequences, result_sequence_names, result_sequence_lengths);
		
		std::string s_result_sequence_names = std::string(result_sequence_names[0]);
		std::string s_sequence_names = std::string(sequence_names[0]);
		
		REQUIRE( result == 0 );
		REQUIRE( num_result_sequences == num_sequences);
		REQUIRE( result_sequences[0][0] == sequences[0][0]);
		REQUIRE( s_result_sequence_names == s_sequence_names);
		REQUIRE( result_sequence_lengths[0] == sequence_lengths[0]);
		
		cudaFreeHost(result_sequences);
		cudaFreeHost(result_sequence_names);
		cudaFreeHost(result_sequence_lengths);
		
		cudaFreeHost(sequences);
		cudaFreeHost(sequence_names);
		cudaFreeHost(sequence_lengths);
		
		std::cerr << std::endl;
		
	}
	
	SECTION("Wrong Size File Data"){
		
		std::cerr << "------TEST 2: enforce same length on FAST5 output ------" << std::endl;
		
		short **sequences;
		char ** sequence_names;
		size_t *sequence_lengths;
	
		cudaMallocHost(&sequences, sizeof(short *)*3); CUERR("Allocating CPU memory for sequence pointers");
		cudaMallocHost(&sequence_names, sizeof(char *)*3); CUERR("Allocating CPU memory for sequence lengths");
		cudaMallocHost(&sequence_lengths, sizeof(size_t)*3); CUERR("Allocating CPU memory for sequence lengths");
		
		int num_sequences = read_tsv_data<short>(filenames[1], sequences, sequence_names, sequence_lengths);
		
		int result = writeFast5Output(stringToChar(series_filename_seq), stringToChar(fast5_output), sequence_names, sequences, sequence_lengths, num_sequences);
	
		
		REQUIRE( result == 1 );

		cudaFreeHost(sequences);
		cudaFreeHost(sequence_names);
		cudaFreeHost(sequence_lengths);
		
		std::cerr << std::endl;
		
	}
	
	SECTION("Wrong Name File Data"){
		
		std::cerr << "------TEST 3: failure on bad file name------" << std::endl;
		
		short **sequences;
		char ** sequence_names;
		size_t *sequence_lengths;
	
		cudaMallocHost(&sequences, sizeof(short *)*3); CUERR("Allocating CPU memory for sequence pointers");
		cudaMallocHost(&sequence_names, sizeof(char *)*3); CUERR("Allocating CPU memory for sequence lengths");
		cudaMallocHost(&sequence_lengths, sizeof(size_t)*3); CUERR("Allocating CPU memory for sequence lengths");
		
		int num_sequences = read_tsv_data<short>(filenames[2], sequences, sequence_names, sequence_lengths);
		
		int result = writeFast5Output(stringToChar(series_filename_seq), stringToChar(fast5_output), sequence_names, sequences, sequence_lengths, num_sequences);
		
		
		REQUIRE( result == 1 );
		
		cudaFreeHost(sequences);
		cudaFreeHost(sequence_names);
		cudaFreeHost(sequence_lengths);
		
		std::cerr << std::endl;
		
	}
	
	free(filenames);
	
}
