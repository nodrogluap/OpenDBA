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

#include "../openDBA.cuh"

char* cur_dir_char = (char*) malloc(FILENAME_MAX);
char* tmp = GetCurrentDir( cur_dir_char, FILENAME_MAX );
std::string current_working_dir(cur_dir_char);

TEST_CASE( " Setup and Run " ){

	std::string s_output_prefix = "openDBA_test";

	int num_series = 2;
	int min_segment_length = 0; // reasonable settings for nanopore RNA dwell time distributions would be 4 (lower to 2 for DNA)
	
	// Both 0 for global
	int use_open_start = 0;
	int use_open_end = 0;
	
	int norm_sequences = 0;
	
	SECTION("Good Data Text float"){
		std::cerr << "------TEST SETUPANDRUN GOOD DATA TEXT FLOAT------" << std::endl;
		
		int read_mode = TEXT_READ_MODE;
		
		// std::string s_seqprefix_filename = current_working_dir + "/good_files/text/good_test_sub.txt";
		// char *seqprefix_filename = (char*)malloc(s_seqprefix_filename.length() + 1);
		// strcpy(seqprefix_filename, s_seqprefix_filename.c_str());
		
		// std::string s_seqprefix_filename = "/dev/null";
		// char *seqprefix_filename = (char*)malloc(s_seqprefix_filename.length() + 1);
		// strcpy(seqprefix_filename, s_seqprefix_filename.c_str());
		char *seqprefix_filename = 0;
		std::string s_series_filename_seq1 = current_working_dir + "/good_files/text/test2/random_short1.txt";
		
		char *series_filename_seq1 = (char*)malloc(s_series_filename_seq1.length() + 1);
		strcpy(series_filename_seq1, s_series_filename_seq1.c_str());
		
		std::string s_series_filename_seq2 = current_working_dir + "/good_files/text/test2/random_short2.txt";
		char *series_filename_seq2= (char*)malloc(s_series_filename_seq2.length() + 1);
		strcpy(series_filename_seq2, s_series_filename_seq2.c_str());
		
		char** series_filenames = (char**)malloc(sizeof(char*)*num_series);
		series_filenames[0] = series_filename_seq1;
		series_filenames[1] = series_filename_seq2;
		
		std::string s_output_prefix_gdti = s_output_prefix + "gdti";
		char *output_prefix = (char*)malloc(s_output_prefix_gdti.length() + 1);
		strcpy(output_prefix, s_output_prefix_gdti.c_str());
		
		// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
		setupAndRun<float>(seqprefix_filename, series_filenames, num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
		
		free(seqprefix_filename);
		free(series_filename_seq1);
		free(series_filename_seq2);
		free(series_filenames);
		free(output_prefix);
		
		std::cerr << std::endl;
		
	}
	/*
	SECTION("Simple Data Text float"){
		std::cerr << "------TEST SETUPANDRUN SIMPLE DATA TEXT FLOAT------" << std::endl;
		
		int read_mode = TEXT_READ_MODE;
		
		// std::string s_seqprefix_filename = current_working_dir + "/good_files/text/good_test_sub.txt";
		// char *seqprefix_filename = (char*)malloc(s_seqprefix_filename.length() + 1);
		// strcpy(seqprefix_filename, s_seqprefix_filename.c_str());
		
		// std::string s_seqprefix_filename = "/dev/null";
		// char *seqprefix_filename = (char*)malloc(s_seqprefix_filename.length() + 1);
		// strcpy(seqprefix_filename, s_seqprefix_filename.c_str());
		char *seqprefix_filename = 0;
		
		std::string s_series_filename_seq1 = current_working_dir + "/good_files/text/test1/simple3_test.txt";
		char *series_filename_seq1 = (char*)malloc(s_series_filename_seq1.length() + 1);
		strcpy(series_filename_seq1, s_series_filename_seq1.c_str());
		
		std::string s_series_filename_seq2 = current_working_dir + "/good_files/text/test1/simple4_test.txt";
		char *series_filename_seq2 = (char*)malloc(s_series_filename_seq2.length() + 1);
		strcpy(series_filename_seq2, s_series_filename_seq2.c_str());
		
		char** series_filenames = (char**)malloc(sizeof(char*)*num_series);
		series_filenames[0] = series_filename_seq1;
		series_filenames[1] = series_filename_seq2;
		
		std::string s_output_prefix_gdti = s_output_prefix + "sdti";
		char *output_prefix = (char*)malloc(s_output_prefix_gdti.length() + 1);
		strcpy(output_prefix, s_output_prefix_gdti.c_str());
		
		// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
		setupAndRun<float>(seqprefix_filename, series_filenames, num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
		
		free(seqprefix_filename);
		free(series_filename_seq1);
		free(series_filename_seq2);
		free(series_filenames);
		free(output_prefix);
		
		std::cerr << std::endl;
		
	}

	SECTION("Good Data Binary uint"){
		std::cerr << "------TEST SETUPANDRUN GOOD DATA BINARY UINT------" << std::endl;

		int read_mode = BINARY_READ_MODE;
		
		std::string s_output_prefix_gdbui = s_output_prefix + "gdbui";
		
		char *output_prefix = (char*)malloc(s_output_prefix_gdbui.length() + 1);
		strcpy(output_prefix, s_output_prefix_gdbui.c_str());
		
		// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
		setupAndRun<unsigned int>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length);
		
		std::cerr << std::endl;
		
	}
	
	
	SECTION("Good Data TSV ulong"){
		std::cerr << "------TEST SETUPANDRUN GOOD DATA TSV ULONG------" << std::endl;
		
		int read_mode = TSV_READ_MODE;
		
		std::string s_output_prefix_gdtul = s_output_prefix + "gdtul";
		
		char *output_prefix = (char*)malloc(s_output_prefix_gdtul.length() + 1);
		strcpy(output_prefix, s_output_prefix_gdtul.c_str());
		
		// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
		setupAndRun<unsigned long long>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length);
		
		std::cerr << std::endl;
		
	}
	
	
	SECTION("Good Data FAST5 float"){
		std::cerr << "------TEST SETUPANDRUN GOOD DATA FAST5 FLOAT------" << std::endl;
		
		int read_mode = FAST5_READ_MODE;
		
		std::string s_output_prefix_gdff = s_output_prefix + "gdff";
		
		char *output_prefix = (char*)malloc(s_output_prefix_gdff.length() + 1);
		strcpy(output_prefix, s_output_prefix_gdff.c_str());
		
		// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
		setupAndRun<float>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length);
		
		std::cerr << std::endl;
		
	}
	
	SECTION("Good Data Text double"){
		std::cerr << "------TEST SETUPANDRUN GOOD DATA TEXT DOUBLE------" << std::endl;

		int read_mode = TEXT_READ_MODE;
		
		std::string s_output_prefix_gdtd = s_output_prefix + "gdtd";
		
		char *output_prefix = (char*)malloc(s_output_prefix_gdtd.length() + 1);
		strcpy(output_prefix, s_output_prefix_gdtd.c_str());
		
		// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
		setupAndRun<double>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length);
		
		std::cerr << std::endl;
		
	}
	*/
	// Following needed to allow cuda-memcheck to detect memory leaks
	cudaDeviceReset(); CUERR("Resetting GPU device");
	
}