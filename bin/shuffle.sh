#!/bin/bash
usage="$(basename "$0") [-h] [-s n] [-i s] [-o s] [-a b]-- 

where:
-h  show this help text
-s  set the seed value (default: 42)
-a  sort (default: false)
-i  set input filepath
-o  set output filepath"

seed=42
sorted=false
while getopts i:o:s:a: flag; do
    case "${flag}" in
        h) echo "$usage"
            exit
            ;;
        i) input=${OPTARG};;
        o) output=${OPTARG};;
        s) seed=${OPTARG};;
        a) sorted=${OPTARG};;
        :) printf "missing argument for -%s\n" "$OPTARG" >&2
            echo "$usage" >&2
            exit 1
            ;;
        \?) printf "illegal option: -%s\n" "$OPTARG" >&2
            echo "$usage" >&2
            exit 1
            ;;
    esac
done
echo "$input $output $seed"
get_seeded_random()
{
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
        </dev/zero 2>/dev/null

    }
if $sorted
then
    sort -o $output $input -S 500M
else
    shuf -o $output <$input --random-source=<(get_seeded_random $seed)
fi
