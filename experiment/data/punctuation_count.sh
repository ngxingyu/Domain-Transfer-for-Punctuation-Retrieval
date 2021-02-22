for split in "dev" "test" "train"
do
for file in /home/nxingyu2/data/open*.$split.csv
do
 echo $file
 sed -E 's/[^[:punct:]]//g;s/(.)/\1x/g' $file  | tr 'x' '\n' | sort | uniq -c | awk '{array[$2]=$1; sum+=$1} END { for (i in array) printf "%-20s %-15d %6.2f%%\n", i, array[i], array[i]/sum*100}' | sort -r -k2,2 -n
done
done