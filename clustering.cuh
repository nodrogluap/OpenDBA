#ifndef __CLUSTERING_CUH
#define __CLUSTERING_CUH

#define ARITH_SERIES_SUM(n) (((n)*(n+1))/2)
// Convenience macro to calculate data row offset in upper right triangle of all vs. all pairwise distances 1D "matrix" representation
#define PAIRWISE_DIST_ROW(i,num_seqs) (ARITH_SERIES_SUM(num_seqs-1)-ARITH_SERIES_SUM(num_seqs - i - 1))
 
/* Iteratively define clusters from the leaves up, using permutation testing to see if the clusters predefined in the provided 'merge' array 
 * (the results of complete linkage) are non-random. */

template<typename T>
__host__
void 
merge_clusters(T **gpu_sequences, size_t *sequence_lengths, size_t num_sequences, T *dtwPairwiseDistances, int *merge, float cluster_p_value, int *memberships, int use_open_start, int use_open_end, cudaStream_t stream){

        for(int i = 0; i < num_sequences; i++){
		memberships[i] = i; // everything starts in its own cluster
                // NB: indices of observables in merge start at 1 (R convention)
                // The hclust convention is a (n-1)*2 matrix for n sequences in a clustering result, where for each merge pair,
                // leaf nodes are indicated with a negative number
                // (m1,m2) = merge[i,] node pair
                int m1 = merge[i];
                int m2 = merge[num_sequences-1+i];
                if (m1 < 0 && m2 < 0) { // both leaf nodes single observables in the complete linkage results
                        // For each leaf node, see if its joining leaf after complete linkage is lower cost than x% of all possible pairings for that leaf
                        // (since we've precomputed all pairwise alignments, we might as well use the data).
                        // Count the distances greater between this leaf and all other non-merge leaves, to find the rank of the
                        // joined leafs within that list to get an exact p-value for this merge
                        // For simplicity, make m1 the earlier index, and for sanity's sake change to zero-based indexing
                        if(m1 < m2){ // note they are negative numbers by R convention, so < is actually > once we flip the signs later
                                int temp = m1;
                                m1 = m2;
                                m2 = temp;
                        }
                        m1 = -m1-1;
                        m2 = -m2-1;
                        //std::cerr << i << " " << sequence_names[m1] << ":" << sequence_names[m1] << std::endl;
                        int dists_smaller = 0;
                        int m1row_index_offset = PAIRWISE_DIST_ROW(m1, num_sequences);
                        T merge_dist = dtwPairwiseDistances[m1row_index_offset+m2-m1-1];
                        // Check the m1 column for all rows before m1 in the upper right triangle of pairwise distances
                        for(int j = 0; j < m1; ++j){
                                if(dtwPairwiseDistances[PAIRWISE_DIST_ROW(j, num_sequences)+m1-j-1] < merge_dist){ // -1 because we aren't storing the diagonal zeroes
                                        dists_smaller++;
                                }
                        }
                        // Check the row for m1
                        for(int j = m1; j < num_sequences - 1; ++j){
                                if(dtwPairwiseDistances[m1row_index_offset+j-m1-1] < merge_dist){
                                        dists_smaller++;
                                }
                        }
                        float p_value = ((float) dists_smaller + 1)/(num_sequences-1);  // There are n-1 possible pairings,
                                                                                        // counted the smaller dists, so picking the most optimistic
                                                                                        // p-value in case there are ties.
                        if(dists_smaller == 0 || p_value < cluster_p_value){

				// Perform DTW, then average the two (no convergence required)
				std::string NO_FILE_OUTPUT = std::string(); //empty
				size_t TWO_SEQS = 2;
				T *medoidSequence = gpu_sequences[m2]; // The convention is to pick the longer of the two. Since we sort by size, m2 is the longer (or same length) one.
				size_t medoidLength = sequence_lengths[m2];
				T *leafSequences[2] = {gpu_sequences[m1], gpu_sequences[m2]};
				size_t leaf_sequence_lengths[2] = {sequence_lengths[m1], sequence_lengths[m2]};

				T *leavesAveragedSequence;
				cudaMallocManaged(leavesAveragedSequence, sizeof(unsigned int)*medoidLength); CUERR("Allocating GPU memory for two leaf mean sequence pileup");

				DBAUpdate(medoidSequence, medoidLength, leafSequences, TWO_SEQS, leaf_sequence_lengths, use_open_start, use_open_end, leavesAveragedSequence, NO_FILE_OUTPUT, stream);

				memberships[m2] = i;
                        }

                }
                // Else it's a leaf connected to an internal node of the tree and we'll need to compare that sequence to the centroid sequence of the
                // inner node (which still needs to be computed).

        }
        std::cerr << std::endl;
        //std::cerr << "Found " << clusterIndicesInMerge.size() << " cluster seed pairs with p-value < " << cluster_p_value << std::endl;

        for(int i = 0; i < num_sequences; i++){
		// merge[i][] contains the merged nodes in step i, and it's a (n-1)x2 matrix (2*(n-1) array)
		// merge[i][j] is negative when the node is an atom
		// For each new cluster we need to generate the centroid
		int m1 = merge[i];
                int m2 = merge[num_sequences-1+i];

		// There are three meaningful cases possible: merging two leaves, merging a single sequence in to an existing cluster, or merging two existing clusters. 
		// Anything else is a stop condition as clustering depends on commutative union operations amongst the members for us.
		if(m1 < 0 && m2 < 0){ // leaf merger, we computed their suitability in the previous loop as indicated by having same membership

			if(memberships[m1] == memberships[m2]){
			}
		}
		else if(m1 < 0 && m2 > 0 || m1 > 0 && m2 < 0){ // merger of either a leaf and a cluster
			// To get a p-value, compare the DTW distance between 1) the leaf and the cluster centroid to 2) the distances between the cluster centroid and all
			// leafs that aren't already part of the centroid (a.k.a. a hypergeometric test)
		}
		else{	// two clusters that have passed the stats tests

		}
	}
}

#endif
