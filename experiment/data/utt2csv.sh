#!/bin/bash

echo "in $1"
echo "out $2"
:> "$2"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# echo 'talk_id,transcript'> "$2"
#echo "filenames, transcript" > opensubtitles.csv
for folder in $1/sw*; do
    for filename in $folder/*.utt; do
        echo $filename
        python $DIR/processutt.py -i $filename -o $2
        #echo "$filename, \"$transcript\"" >> opensubtitles.csv
        # break 0
    done
done
