#!/bin/bash
usage="$(basename "$0") [-h] [-s n] [-i s] [-o s] [-a b] [-m s] [-t s]-- 

where:
-h  show this help text
-s  set the seed value (default: 42)
-a  sort (default: false)
-i  set input filepath
-o  set output filepath
-m  set memory limit
-t  set tmp dir"

seed=42
sorted=false
memory="500M"
tmpdir="~/data/tmp/"
while getopts i:o:s:a:m:t: flag; do
    case "${flag}" in
        h) echo "$usage"
            exit
            ;;
        i) input=${OPTARG};;
        o) output=${OPTARG};;
        s) seed=${OPTARG};;
        a) sorted=${OPTARG};;
        m) memory=${OPTARG};;
        t) tmpdir=${OPTARG};;
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
    env TMPDIR=$tmpdir sort -o $output $input -S $memory
else
    shuf -o $output <$input --random-source=<(get_seeded_random $seed)
fi
