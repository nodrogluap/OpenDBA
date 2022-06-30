#!/bin/sh
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <length of reference RNA sequence> <OpenDBA cluster file prefix>" >&2
    exit 2
fi

trap 'rm -f "$TMPFILE"' EXIT

TMPFILE=$(mktemp) || exit 1
perl -ane 'push @{$means{$F[2]}}, $F[1] unless $F[4] eq "OPEN_RIGHT" or $#F < 4;END{for (sort {$b <=> $a} keys %means){print '$1'-$_,"\t", join(",", @{$means{$_}}),"\n"}}' $2.path*.txt > $TMPFILE
`dirname $0`/diptest.R $TMPFILE > $2.multimodal.diptest.txt
`dirname $0`/kde_smoothing_plus_excess_mass.R $TMPFILE > $2.multimodal.kde_smoothing_plus_excess_mass.txt
