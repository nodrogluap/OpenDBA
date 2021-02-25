#/usr/bin/env bash

HDF5_PLUGIN_PATH=`dirname $0`/vendor/plugins/vbz_compression/build/bin `dirname $0`/openDBA $*
