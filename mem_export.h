#ifndef MEM_EXPORT_H
#define MEM_EXPORT_H

struct dtw_result{
	char *sequence_name;
	int alignment_length;
	int *sequence_index;
	int *centroid_index;
	char *moves;
} dtw_result;

template <class T> 
struct dba_result{
	int num_sequences;
	T *centroid_sequence;
	int centroid_sequence_length;
	struct dtw_result *seq_centroid_alignment;
};

#endif
