#!/bin/bash

echo "in $1"
echo "out $2"
:> "$2"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# echo 'talk_id,transcript'> "$2"
#echo "filenames, transcript" > opensubtitles.csv
for folder in $1/*; do
    for filename in $folder/*.dff; do
        echo $filename
        python $DIR/processdff.py -i $filename -o $2
        #echo "$filename, \"$transcript\"" >> opensubtitles.csv
        # break 0
    done
done
