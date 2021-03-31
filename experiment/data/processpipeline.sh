#  gdown https://drive.google.com/uc?id=1-Cpoy9ms5Jizu76n50tXtduzyBxxfeWx
if [ "$1" == "-h" ] ; then
    echo "Usage: `basename $0` [-h] inputfilename, chunksize, split 1, split 2, split 3, max_len, degree, threads"
    exit 0
fi

python ~/project/processcsv.py -i $1 -o "${1%.csv}_processed.csv" -c $2
bash ~/project/experiment/data/processandsplit.sh "${1%.csv}_processed.csv" $3 $4 $5
python ~/project/text2aligned.py -i "${1%.csv}_processed" -s 'test' -m $6 -o 0 -d $7 -t $8
python ~/project/text2aligned.py -i "${1%.csv}_processed" -s 'dev' -m $6 -o 0 -d $7 -t $8
python ~/project/text2aligned.py -i "${1%.csv}_processed" -s 'train' -m $6 -o 0 -d $7 -t $8
