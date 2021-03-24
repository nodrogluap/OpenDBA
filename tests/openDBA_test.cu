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
#include "../cpu_utils.hpp"

#include "test_utils.cuh"

char* cur_dir_char = (char*) malloc(FILENAME_MAX);
char* tmp = GetCurrentDir( cur_dir_char, FILENAME_MAX );
std::string current_working_dir(cur_dir_char);

TEST_CASE( " Setup and Run " ){

	std::string s_output_prefix = "openDBA_test_";
	
	int series_buff_size = 3;

	int min_segment_length = 0; // disable segmentation
	
	// Both 0 for global
	int use_open_start = 0;
	int use_open_end = 0;
	
	int norm_sequences = 0;
	int read_mode = TEXT_READ_MODE;
	
	char *seqprefix_filename = 0;
	
	std::string s_series_filename_seq1 = current_working_dir + "/good_files/text/test2/random_short1.txt";
	char *series_filename_seq1 = (char*)malloc(s_series_filename_seq1.length() + 1);
	strcpy(series_filename_seq1, s_series_filename_seq1.c_str());
	
	std::string s_series_filename_seq2 = current_working_dir + "/good_files/text/test2/random_short2.txt";
	char *series_filename_seq2= (char*)malloc(s_series_filename_seq2.length() + 1);
	strcpy(series_filename_seq2, s_series_filename_seq2.c_str());
	
	std::string s_series_filename_seq3 = current_working_dir + "/good_files/text/test2/random_short3.txt";
	char *series_filename_seq3= (char*)malloc(s_series_filename_seq3.length() + 1);
	strcpy(series_filename_seq3, s_series_filename_seq3.c_str());
	
	char** series_filenames = (char**)malloc(sizeof(char*)*series_buff_size);
	
	SECTION("Good Same Data"){
		std::cerr << "------TEST SETUPANDRUN SAME DATA------" << std::endl;
		
		int num_series = 2;
		
		series_filenames[0] = series_filename_seq1;
		series_filenames[1] = series_filename_seq1;
		
		std::string s_output_prefix_gdti = s_output_prefix + "gsd";
		char *output_prefix = (char*)malloc(s_output_prefix_gdti.length() + 1);
		strcpy(output_prefix, s_output_prefix_gdti.c_str());
		
		// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
		setupAndRun<float>(seqprefix_filename, newCharArraysDeepCopy(series_filenames, num_series), num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
		
		std::string s_avg_txt_file = s_output_prefix_gdti + ".avg.txt";
		char *avg_txt_file = (char*)malloc(s_avg_txt_file.length() + 1);
		strcpy(avg_txt_file, s_avg_txt_file.c_str());
		
		float *avg_return_values;
		size_t num_avg_return_values;
		
		read_text_data<float>(avg_txt_file, &avg_return_values, &num_avg_return_values);
		
		REQUIRE( num_avg_return_values == 10 );
		REQUIRE( avg_return_values[0] == 1 );
		REQUIRE( round_to_three(avg_return_values[1]) == 0.895f );
		REQUIRE( round_to_three(avg_return_values[2]) == 0.8f );
		REQUIRE( round_to_three(avg_return_values[3]) == 0.644f );
		REQUIRE( round_to_three(avg_return_values[4]) == 0.586f );
		REQUIRE( round_to_three(avg_return_values[5]) == 0.488f );
		REQUIRE( round_to_three(avg_return_values[6]) == 0.357f );
		REQUIRE( round_to_three(avg_return_values[7]) == 0.287f );
		REQUIRE( round_to_three(avg_return_values[8]) == 0.139f );
		REQUIRE( round_to_three(avg_return_values[9]) == 0.017f );
		
		free(output_prefix);
		free(avg_txt_file);
		
		std::cerr << std::endl;
		
	}
	
	SECTION("Good Different Data"){
		std::cerr << "------TEST SETUPANDRUN DIFFERENT DATA------" << std::endl;
		
		int num_series = 2;
		
		series_filenames[0] = series_filename_seq1;
		series_filenames[1] = series_filename_seq2;
		
		std::string s_output_prefix_gdti = s_output_prefix + "gdd";
		char *output_prefix = (char*)malloc(s_output_prefix_gdti.length() + 1);
		strcpy(output_prefix, s_output_prefix_gdti.c_str());
		
		// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
		setupAndRun<float>(seqprefix_filename, newCharArraysDeepCopy(series_filenames, num_series), num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
		
		std::string s_avg_txt_file = s_output_prefix_gdti + ".avg.txt";
		char *avg_txt_file = (char*)malloc(s_avg_txt_file.length() + 1);
		strcpy(avg_txt_file, s_avg_txt_file.c_str());
		
		float *avg_return_values;
		size_t num_avg_return_values;
		
		read_text_data<float>(avg_txt_file, &avg_return_values, &num_avg_return_values);
		
		REQUIRE( num_avg_return_values == 10 );
		REQUIRE( avg_return_values[0] == 1 );
		REQUIRE( round_to_three(avg_return_values[1]) == 0.895f );
		REQUIRE( round_to_three(avg_return_values[2]) == 0.796f );
		REQUIRE( round_to_three(avg_return_values[3]) == 0.665f );
		REQUIRE( round_to_three(avg_return_values[4]) == 0.59f );
		REQUIRE( round_to_three(avg_return_values[5]) == 0.456f );
		REQUIRE( round_to_three(avg_return_values[6]) == 0.347f );
		REQUIRE( round_to_three(avg_return_values[7]) == 0.265f );
		REQUIRE( round_to_three(avg_return_values[8]) == 0.137f );
		REQUIRE( round_to_three(avg_return_values[9]) == 0.05f );
		
		free(output_prefix);
		free(avg_txt_file);
		
		std::cerr << std::endl;
		
	}
	
	SECTION("Three Different Files"){
		std::cerr << "------TEST SETUPANDRUN THREE DIFFERENT FILES------" << std::endl;
		
		int num_series = 3;
		
		series_filenames[0] = series_filename_seq1;
		series_filenames[1] = series_filename_seq2;
		series_filenames[2] = series_filename_seq3;
		
		std::string s_output_prefix_gdti = s_output_prefix + "gtdd";
		char *output_prefix = (char*)malloc(s_output_prefix_gdti.length() + 1);
		strcpy(output_prefix, s_output_prefix_gdti.c_str());
		
		// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
		setupAndRun<float>(seqprefix_filename, newCharArraysDeepCopy(series_filenames, num_series), num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
		
		std::string s_avg_txt_file = s_output_prefix_gdti + ".avg.txt";
		char *avg_txt_file = (char*)malloc(s_avg_txt_file.length() + 1);
		strcpy(avg_txt_file, s_avg_txt_file.c_str());
		
		float *avg_return_values;
		size_t num_avg_return_values;
		
		read_text_data<float>(avg_txt_file, &avg_return_values, &num_avg_return_values);
		
		REQUIRE( num_avg_return_values == 10 );
		REQUIRE( round_to_three(avg_return_values[0]) == 0.972f );
		REQUIRE( round_to_three(avg_return_values[1]) == 0.901f );
		REQUIRE( round_to_three(avg_return_values[2]) == 0.814f );
		REQUIRE( round_to_three(avg_return_values[3]) == 0.681f );
		REQUIRE( round_to_three(avg_return_values[4]) == 0.56f );
		REQUIRE( round_to_three(avg_return_values[5]) == 0.455f );
		REQUIRE( round_to_three(avg_return_values[6]) == 0.362f );
		REQUIRE( round_to_three(avg_return_values[7]) == 0.244f );
		REQUIRE( round_to_three(avg_return_values[8]) == 0.128f );
		REQUIRE( round_to_three(avg_return_values[9]) == 0.056f );
		
		free(output_prefix);
		free(avg_txt_file);
		
		std::cerr << std::endl;
		
	}
	
	SECTION("Good Different Truncated Data"){
		std::cerr << "------TEST SETUPANDRUN DIFFERENT TRUNCATED DATA------" << std::endl;
		
		int num_series = 2;
		
		std::string s_series_filename_seq2_trun = current_working_dir + "/good_files/text/test2/random_trun_short2.txt";
		char *series_filename_seq2_trun = (char*)malloc(s_series_filename_seq2_trun.length() + 1);
		strcpy(series_filename_seq2_trun, s_series_filename_seq2_trun.c_str());
		
		series_filenames[0] = series_filename_seq1;
		series_filenames[1] = series_filename_seq2_trun;
		
		use_open_end = 1;
		
		std::string s_output_prefix_gdti = s_output_prefix + "gdtd";
		char *output_prefix = (char*)malloc(s_output_prefix_gdti.length() + 1);
		strcpy(output_prefix, s_output_prefix_gdti.c_str());
		
		// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
		setupAndRun<float>(seqprefix_filename, newCharArraysDeepCopy(series_filenames, num_series), num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
		
		std::string s_avg_txt_file = s_output_prefix_gdti + ".avg.txt";
		char *avg_txt_file = (char*)malloc(s_avg_txt_file.length() + 1);
		strcpy(avg_txt_file, s_avg_txt_file.c_str());
		
		float *avg_return_values;
		size_t num_avg_return_values;
		
		read_text_data<float>(avg_txt_file, &avg_return_values, &num_avg_return_values);
		
		REQUIRE( num_avg_return_values == 10 );
		REQUIRE( avg_return_values[0] == 1 );
		REQUIRE( round_to_three(avg_return_values[1]) == 0.895f );
		REQUIRE( round_to_three(avg_return_values[2]) == 0.796f );
		REQUIRE( round_to_three(avg_return_values[3]) == 0.665f );
		REQUIRE( round_to_three(avg_return_values[4]) == 0.59f );		
		REQUIRE( round_to_three(avg_return_values[5]) == 0.488f );
		REQUIRE( round_to_three(avg_return_values[6]) == 0.357f );
		REQUIRE( round_to_three(avg_return_values[7]) == 0.287f );
		REQUIRE( round_to_three(avg_return_values[8]) == 0.139f );
		REQUIRE( round_to_three(avg_return_values[9]) == 0.017f );
		
		free(series_filename_seq2_trun);
		
		free(output_prefix);
		free(avg_txt_file);
		
		std::cerr << std::endl;
		
	}
	
	SECTION("Good Large Data"){
		std::cerr << "------TEST SETUPANDRUN LARGE DATA------" << std::endl;
		
		int num_series = 2;
		
		std::string s_series_filename_seq1_large= current_working_dir + "/good_files/text/test2/random_large0.txt";
		char *series_filename_seq1_large= (char*)malloc(s_series_filename_seq1_large.length() + 1);
		strcpy(series_filename_seq1_large, s_series_filename_seq1_large.c_str());
		
		std::string s_series_filename_seq2_large = current_working_dir + "/good_files/text/test2/random_large1.txt";
		char *series_filename_seq2_large = (char*)malloc(s_series_filename_seq2_large.length() + 1);
		strcpy(series_filename_seq2_large, s_series_filename_seq2_large.c_str());
		
		series_filenames[0] = series_filename_seq1_large;
		series_filenames[1] = series_filename_seq2_large;
		
		std::string s_output_prefix_gdti = s_output_prefix + "ld";
		char *output_prefix = (char*)malloc(s_output_prefix_gdti.length() + 1);
		strcpy(output_prefix, s_output_prefix_gdti.c_str());
		
		// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
		setupAndRun<float>(seqprefix_filename, newCharArraysDeepCopy(series_filenames, num_series), num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences);
		
		std::string s_avg_txt_file = s_output_prefix_gdti + ".avg.txt";
		char *avg_txt_file = (char*)malloc(s_avg_txt_file.length() + 1);
		strcpy(avg_txt_file, s_avg_txt_file.c_str());
		
		float *avg_return_values;
		size_t num_avg_return_values;
		
		read_text_data<float>(avg_txt_file, &avg_return_values, &num_avg_return_values);
		
		REQUIRE( num_avg_return_values == 2048 );
		REQUIRE( avg_return_values[0] == 1 );
		REQUIRE( round_to_three(avg_return_values[1]) == 0.997f );
		REQUIRE( round_to_three(avg_return_values[2]) == 1 );
		REQUIRE( round_to_three(avg_return_values[3]) == 0.959f );
		// REQUIRE( round_to_three(avg_return_values[4]) == 0.59f );		
		// REQUIRE( round_to_three(avg_return_values[5]) == 0.488f );
		// REQUIRE( round_to_three(avg_return_values[6]) == 0.357f );
		// REQUIRE( round_to_three(avg_return_values[7]) == 0.287f );
		// REQUIRE( round_to_three(avg_return_values[8]) == 0.139f );
		REQUIRE( round_to_three(avg_return_values[2047]) == -0.457f );
		
		free(series_filename_seq1_large);
		free(series_filename_seq2_large);
		
		free(output_prefix);
		free(avg_txt_file);
		
		std::cerr << std::endl;
		
	}
	
	free(series_filename_seq1);
	free(series_filename_seq2);
	free(series_filename_seq3);
	free(series_filenames);
	
	// Following needed to allow cuda-memcheck to detect memory leaks
	cudaDeviceReset(); CUERR("Resetting GPU device");
	
}
