#!/bin/bash
touch opensubtitles.csv
> ~/data/opensubtitles.csv
#echo "filenames, transcript" > opensubtitles.csv
for year in ~/data/OpenSubtitles/raw/en/*; do
    for movie_id in $year/*; do
        for filename in $movie_id/*.xml; do
            echo $filename
            python ~/project/processxml.py -i $filename -o ~/data/opensubtitles.csv
            #echo "$filename, \"$transcript\"" >> opensubtitles.csv
            break 1
        done
    done
done
