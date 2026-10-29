#!/bin/sh
# mean early/late per band (power mean over clicks) of an assess `still` output
grep '^ *click ' "$1" | awk '{for(i=4;i<=6;i++) e[i]+=10^($i/10); for(i=8;i<=10;i++) l[i]+=10^($i/10); n++} END {printf "early %6.1f %6.1f %6.1f   late %6.1f %6.1f %6.1f\n", 10*log(e[4]/n)/log(10), 10*log(e[5]/n)/log(10), 10*log(e[6]/n)/log(10), 10*log(l[8]/n)/log(10), 10*log(l[9]/n)/log(10), 10*log(l[10]/n)/log(10)}'
