#Download zipfile from https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset
mkdir ~/data/
cd ~/data/
#wget -O TedTalks.zip http://opus.nlpl.eu/download.php?f=TedTalks/v1/raw/en.zip
wget -O OpenSubtitles.zip http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2016/raw/en.zip
#unzip TedTalks.zip
unzip OpenSubtitles.zip
bash ~/project/xml2csv.sh -i ./OpenSubtitles -o ./OpenSubtitles.csv
