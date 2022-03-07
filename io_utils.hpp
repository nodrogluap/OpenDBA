#ifndef __io_utils_hpp_included
#define __io_utils_hpp_included

// For definition of DTW moves NIL, RIGHT, UP...
#include "dtw.hpp"
#include "exit_codes.hpp"

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
int writeDTWPath(unsigned char *cpu_pathMatrix, const char *filename, T *gpu_seq, char *cpu_seqname, size_t gpu_seq_len, T *cpu_centroid, size_t cpu_centroid_len, size_t num_columns, size_t num_rows, size_t pathPitch, int flip_seq_order){
	std::ofstream path(filename);
	if(!path.is_open()){
        	std::cerr << "Cannot write to " << filename << std::endl;
        	return CANNOT_WRITE_DTW_PATH;
	}
	path << cpu_seqname << std::endl;

	T *cpu_seq;
	cudaMallocHost(&cpu_seq, sizeof(T)*gpu_seq_len); CUERR("Allocating CPU memory for query seq in DTW path printing");
	cudaMemcpy(cpu_seq, gpu_seq, sizeof(T)*gpu_seq_len, cudaMemcpyDeviceToHost); CUERR("Copying incoming GPU query to CPU in DTW path printing");

	// moveI and moveJ are defined device-side in dtw.hpp, but we are host side so we need to replicate
	int moveI[] = { -1, -1, 0, -1, 0 };
	int moveJ[] = { -1, -1, -1, 0, -1 };
	int j = num_columns - 1;
	int i = num_rows - 1;
	unsigned char move = cpu_pathMatrix[pitchedCoord(j,i,pathPitch)];
	while (move != NIL && move != NIL_OPEN_RIGHT) {
        	if(flip_seq_order){
                	path << j << "\t" << cpu_seq[j] << "\t" << i << "\t" << cpu_centroid[i] << "\t" << (move == DIAGONAL ? "DIAG" : (move == RIGHT ? "RIGHT" : (move == UP ? "UP" : (move==OPEN_RIGHT ?  "OPEN_RIGHT" : (move ==NIL ? "NIL" : "?")))))<< std::endl;
        	}
        	else{
                	path << i << "\t" << cpu_seq[i] << "\t" << j << "\t" << cpu_centroid[j] << "\t" << (move == DIAGONAL ? "DIAG" : (move == RIGHT ? "RIGHT" : (move == UP ? "UP" : (move==OPEN_RIGHT ?  "OPEN_RIGHT" : (move ==NIL ? "NIL" : "?")))))<< std::endl;
        	}
        	i += moveI[move];
        	j += moveJ[move];
        	move = cpu_pathMatrix[pitchedCoord(j,i,pathPitch)];
	}
	path.close();
	cudaFreeHost(cpu_seq); CUERR("Freeing CPU memory for query seq in DTW path printing");
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
