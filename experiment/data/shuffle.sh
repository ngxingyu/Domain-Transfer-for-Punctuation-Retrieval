#!/bin/bash
usage="$(basename "$0") [-h] [-s n] [-i s] [-o s] [-a b] [-m s]-- 

where:
-h  show this help text
-s  set the seed value (default: 42)
-a  sort (default: false)
-i  set input filepath
-o  set output filepath
-m  set memory limit"

seed=42
sorted=false
memory="500M"
while getopts i:o:s:a:m: flag; do
    case "${flag}" in
        h) echo "$usage"
            exit
            ;;
        i) input=${OPTARG};;
        o) output=${OPTARG};;
        s) seed=${OPTARG};;
        a) sorted=${OPTARG};;
        m) memory=${OPTARG};;
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
    # env TMPDIR=~/data/tmp
    sort -o $output $input -S $memory
else
    shuf -o $output <$input --random-source=<(get_seeded_random $seed)
fi
