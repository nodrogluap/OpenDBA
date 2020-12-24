#ifndef __TEST_UTILS
#define __TEST_UTILS

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