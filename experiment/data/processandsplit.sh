if [ "$1" == "-h" ] ; then
    echo "Usage: `basename $0` [-h] inputfilename, split 1, split 2, split 3 ratio"
    exit 0
fi

echo "in $1"
echo "out $1 _ split .csv"
bash ~/project/bin/shuffle.sh -i $1 -o $1 -a true
# Do not shuffle to maintain chronological order
# bash ~/project/bin/shuffle.sh -i $1 -o $1 -a false -s 42
bash ~/project/bin/percsplit.sh -f "${1%.csv}" $1 $2 $3 $4
mv "${1%.csv}00" "${1%.csv}.train.csv"
mv "${1%.csv}01" "${1%.csv}.dev.csv"
mv "${1%.csv}02" "${1%.csv}.test.csv"
