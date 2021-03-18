#  gdown https://drive.google.com/uc?id=1-Cpoy9ms5Jizu76n50tXtduzyBxxfeWx
python ~/project/processcsv.py -i ./opensubtitles.csv -o ./open_subtitles_processed.csv -c 2000
bash ~/project/bin/processandsplit.sh ./open_subtitles_processed.csv 8 1 1
python ~/project/text2aligned.py -i ./open_subtitles_processed -s 'test' -m 256 -o 0 -d 0 -t 2
python ~/project/text2aligned.py -i ./open_subtitles_processed -s 'dev' -m 256 -o 0 -d 0 -t 2
python ~/project/text2aligned.py -i ./open_subtitles_processed -s 'train' -m 256 -o 0 -d 0 -t 2
