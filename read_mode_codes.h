#ifndef READ_MODE_CODES_H
#define READ_MODE_CODES_H

#define TEXT_READ_MODE 0
#define BINARY_READ_MODE 1
#define TSV_READ_MODE 2
#if HDF5_SUPPORTED == 1
#define FAST5_READ_MODE 3
#endif

#endif