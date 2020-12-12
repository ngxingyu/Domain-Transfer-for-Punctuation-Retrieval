#!/bin/bash

echo "in $1"
echo "out $2"
touch "$2"
echo 'talk_id,transcript'> "$2"
#echo "filenames, transcript" > opensubtitles.csv
for year in $1/raw/en/*; do
    for movie_id in $year/*; do
        for filename in $movie_id/*.xml; do
            echo $filename
            python ./processxml.py -i $filename -o $2
            #echo "$filename, \"$transcript\"" >> opensubtitles.csv
            break 1
        done
    done
done
