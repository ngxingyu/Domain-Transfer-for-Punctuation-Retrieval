#!/bin/bash
while getopts i:o: flag
do
    case "${flag}" in
        i) input;;
        o) output;;
    esac
done

touch $output
> $output
#echo "filenames, transcript" > opensubtitles.csv
for year in $input/raw/en/*; do
    for movie_id in $year/*; do
        for filename in $movie_id/*.xml; do
            echo $filename
            python ~/project/processxml.py -i $filename -o $output
            #echo "$filename, \"$transcript\"" >> opensubtitles.csv
            break 1
        done
    done
done
