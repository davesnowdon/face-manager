#!/usr/bin/env bash
# Run the manager benchmark suite for all test videos

EXE=./manager-benchmark

# Set of test videos
declare -a videoFile=("../test-data/emptyroom1.mp4"
                      "../test-data/emptyroom2.mp4"
                      "../test-data/lightingchange1.mp4"
                      "../test-data/lightingchange2.mp4"
                      "../test-data/movinghuman1.mp4"
                      "../test-data/movinghuman2.mp4"
                      "../test-data/movinghuman3.mp4"
                      "../test-data/movinghuman4.mp4"
                      "../test-data/stillhuman1.mp4"
                      "../test-data/street1.mp4"
                      "../test-data/street2.mp4"
                      "../test-data/street3.mp4"
                      "../test-data/street4.mp4"
                      "../test-data/street5.mp4"
                      "../test-data/multi-face.mov")

ARCH=$(uname -m)
RESULT_FILE="manager-benchmark-results-${ARCH}.csv"
TMP_FILE=$(mktemp /tmp/benchmarks.XXXXXX)
echo $TMP_FILE

## now loop through the above array
for videoFile in "${videoFile[@]}"
do
   $EXE "$videoFile" 1
done > $TMP_FILE

# write header
grep --max-count=1 'File,' < $TMP_FILE >> $RESULT_FILE

# write data
grep 'End:' < $TMP_FILE | sed 's/End: //g' | sed 's/..\/test-data\///g' >> $RESULT_FILE

echo "Results in ${RESULT_FILE}"
#rm $TMP_FILE