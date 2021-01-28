#!/bin/bash

usage () {
    printf '%s\n' "${0##*/} [-ks] [-f prefix] [-n number] file arg1..." >&2
}

# Collect csplit options
while getopts "ksf:n:" opt; do
    case "$opt" in
        k|s) args+=(-"$opt") ;;           # k: no remove on error, s: silent
        f|n) args+=(-"$opt" "$OPTARG") ;; # f: filename prefix, n: digits in number
        *) usage; exit 1 ;;
    esac
done
shift $(( OPTIND - 1 ))

fname=$1
shift
ratios=("$@")

len=$(wc -l < "$fname")

# Sum of ratios and array of cumulative ratios
for ratio in "${ratios[@]}"; do
    (( total += ratio ))
    cumsums+=("$total")
done

# Don't need the last element
unset cumsums[-1]

# Array of numbers of first line in each split file
for sum in "${cumsums[@]}"; do
    linenums+=( $(( sum * len / total + 1 )) )
done

csplit "${args[@]}" "$fname" "${linenums[@]}"
