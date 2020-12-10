#!/bin/bash
touch opensubtitles.csv
> opensubtitles.csv
#echo "filenames, transcript" > opensubtitles.csv
for year in ./OpenSubtitles/raw/en/*; do
    for movie_id in $year/*; do
        for filename in $movie_id/*.xml; do
            echo $filename
            python ./processxml.py -i $filename
            #echo "$filename, \"$transcript\"" >> opensubtitles.csv
            break 1
        done
    done
done
