#!/bin/bash

echo "in $1"
echo "out $2"
touch "$2"
echo 'talk_id,transcript'> "$2"
#echo "filenames, transcript" > opensubtitles.csv
for folder in $1/*; do
    for filename in $folder/*; do
        echo $filename
        python ./processdff.py -i $filename -o $2
        #echo "$filename, \"$transcript\"" >> opensubtitles.csv
        # break 0
    done
done
