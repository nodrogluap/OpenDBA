#ifndef __TEST_UTILS
#define __TEST_UTILS

#include <string.h>

#define ARRAYSIZE(a) \
  ((sizeof(a) / sizeof(*(a))) / \
  static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))

char ** newCharArraysDeepCopy(char **orig, size_t num_items){
	char **cudaHostDeepCopy;
	cudaMallocHost(&cudaHostDeepCopy, sizeof(char *)*num_items);
	for(int i = 0; i < num_items; i++){
		cudaMallocHost(&cudaHostDeepCopy[i], sizeof(char)*strlen(orig[i]));
		cudaMemcpy(cudaHostDeepCopy[i], orig[i], sizeof(char)*strlen(orig[i]), cudaMemcpyHostToHost);
	}
	return cudaHostDeepCopy;
}

float round_to_three(float var);

// Modified from https://www.geeksforgeeks.org/rounding-floating-point-number-two-decimal-places-c-c/
float round_to_three(float var){
	    // we use array of chars to store number 
    // as a string. 
    char str[40];  
  
    // Print in string the value of var  
    // with two decimal point 
    sprintf(str, "%.3f", var); 
  
    // scan string value in var  
    sscanf(str, "%f", &var);  
  
    return var;  
}
#endif
