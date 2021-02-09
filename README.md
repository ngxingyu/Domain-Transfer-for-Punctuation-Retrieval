# ASR
## Datasources
The chosen datasources for this project are:
1. [TED - Ultimate Dataset | Kaggle](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset)-A collection of 4005 TED talks.
2. [Untokenised Corpus files for Opensubtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php) - Select the rightmost column language ID, in my case en.
3. BookCorpusOpen from Huggingface Datasets, a precompiled collection of 17868 books.


## Preprocessing
The following steps are taken for preprocessing the data to a useable form.

1. (Important in the case of TED dataset) Remove speaker tags i.e. Narrator: ... which will not be spoken.
2. Remove tags that are added for better readability, i.e. sound effects (Applause) etc.
3. Identify spoken text that are within square or round brackets and remove the brackets.
4. Remove music lyrics i.e. ♫ Lyrics ♫ 
as a large portion of these do not contain appropriate punctuation information, and removing all of this only reduced the Ted talk corpus from 7156353 to 7140657 words.
5. Remove empty matching tags- square brackets, parentheses, single/double quotes. For single/double quotes, it may be used when training for retrieval of the quotes but I'll look into it again when I'm working on that.
6. Convert ellipsis to the unicode version.
7. Removing Non-sentence punctuation (All punctuation that are non-readable (@$#%&^+•=€²£¥) and not (.?!,;:-–—)).
8. Replace en-dash with hyphen (Since it can easily be recovered with a regex and much of it is used wrongly.)
8. Combining repeated patterns of punctuation, i.e. [. .] to [.], [!!!!] to [!].
9. Replace (no.).(no.) with (no.)<space>point<space>(no.), the common pronounciation of the decimal point.
10. Remove excess whitespaces
11. Remove examples with length < 10 words
12. Random shuffle with seed 42
13. Perform train dev test split of 0.9 0.1 0.1.

### Punctuation proportion
For TED corpus
Train | Val | Test
--- | --- | ---
. 325322    0.394   | 42673 0.3973      | 40906 0.3953
? 29249     0.0354  | 3779  0.03519     | 3746  0.03620
! 2618      0.00317 | 355   0.003305    | 304   0.002938
, 394500    0.478   | 50863 0.4736      | 49212 0.4756
; 4633      0.00561 | 651   0.006062    | 562   0.005432
: 10138     0.0122  | 1366  0.01272     | 1308  0.01264
\- 30341     0.0367  | 3966  0.03693     | 3757  0.03631
— 26402     0.0320  | 3523  0.03280     | 3450  0.03334
… 1715      0.00207 | 206   0.001918    | 214   0.002068
words 5842593       | 757511            | 733686

For subtitles corpus
Train | Val | Test
--- | --- | ---
. 47443035  0.44392753421469083     | 5921757 0.4430048481920479    | 5921304 0.4422626972770535
? 13250829  0.12398886041482206     | 1660096 0.12419127911939411   | 1670946 0.12480309826421737
! 6519047   0.0609991426589736      | 813038 0.06082312660995144    | 834767 0.06234881793231256
, 24551760  0.22973239965425646     | 3062385 0.22909609462708524   | 3063118 0.2287845428570959
; 57285     0.0005360194346227758   | 8073 0.0006039386856729181    | 7344 0.0005485239820152251
: 374276    0.003502124638437183    | 57019 0.004265574125899185    | 44933 0.0033560495756930976
\- 10479724  0.09805945244798349     | 1321067 0.09882862227992877   | 1324557 0.09893127451608667
— 9564      8.949096399986432e-05   | 591 4.421253105818092e-05     | 1210 9.037500248344532e-05
… 4185605   0.03916497557221373     | 523225 0.03914230382896229    | 520479 0.03887462059304226
words 419201886                     | 52422785                      | 52341671

## Processing part-2
The punctuation to be classified are as follows: {1: '!', 2: ',', 3: '-', 4: '.', 5: ':', 6: ';', 7: '?', 8: '—', 9: '…'}
8 is the emdash.
There are occurences of consecutive punctuation. This includes: 
1. ., : period after abbreviation or initial
2. ?, or !— etc. where the first punctuation applies to a local scope and the 2nd applies to a larger context.
3. anomalies i.e. ?! or !! or even a hyphen leading the next sentence
In most cases, it makes more sense to classify the punctuations from right to left, so I will append punctuations to the previous word and predict the punctuation at the top of the stack.:

The process of converting continuous text is as follows:
1. Taking the text and degree, split the text into 2 lists - the first being a list of words, and the second being a list of previously classified punctuations or spaces dividing the text. (i.e. when degree is 0, the 2nd list contains all empty strings.)
2. Intialize 2 new lists, a and b. Process both lists alternately beginning with the words list, identifying the trailing punctuation and stripping all punctuations from the tail. The word will be appended to a and the id of the punctuation identified will be appended to b.
3. 

# Preprocessing commands
'''console
bash ~/project/get-data.sh

python ~/project/processcsv.py -i ~/data/ted_talks_en.csv -o ~/data/ted_talks_processed.csv -c 2000

bash ~/project/bin/processandsplit.sh ./ted_talks_processed.csv 8 1 1

python ~/project/text2aligned.py -i ./ted_talks_processed -d 0 -c 2000 #without any split or filetype

python ~/project/processcsv.py -i ~/data/open_subtitles.csv -o ~/data/open_subtitles_processed.csv -c 2000

bash ~/project/bin/processandsplit.sh ./open_subtitles_processed.csv 8 1 1

python ~/project/text2aligned.py -i ./open_subtitles-processed -d 0 -c 2000

## python installs
pip install conda
pip install nemo_toolkit[all]==1.0.0b2



## Log for 26/1/2020

Found a bug in regex pattern: A-z also includes punctuation characters, use A-Za-z instead.
Worked on creating the model in python instead of ipynb.
python -m pip install git+https://github.com/gruns/icecream.git


## Log for 27/1/2020

Use code-server
```console
user@instance:~$ fuser -k 9999/tcp
user@instance:~$ code-server --bind-addr 127.0.0.1:9999 --auth none &
```

To convert all to lowercse.
To strip leading before first Uppercase, after last sentence punctuation.

Repeated starts are possible i.e. similar show but different episodes. Perhaps better to remove the shuffling and just split by order? yes. I'll do this instead.

Some of the regexes are flawed, to check if spare time?

Convert from huggingface load dataset which loads all to memory to pandas chunking map save.


## Log for 28/1/2020

Found an arabic character in one of the texts "Co ِ perative" which broke the tokenizer parsing. To go through the preprocessing step in greater detail now.


## Log for 2/2/2020

Converted torch Dataset into IterableDataset with chunks, for faster loading. Each batch features a ConcatDataset looking at all children datasets which are cycled using itertools.cycle. They run until the largest batch size is fully covered.

To implement:
Random shuffling of csv dataset each batch.
Look at effectiveness of Dice Loss and possible hyperparameters which can improve its F score.
Evaluate the effectiveness of smaller models 

Git issues: 
```
git filter-branch -f --index-filter 'git rm -rf --cached --ignore-unmatch ./experiment/nemo_experiments/Punctuation_with_Domain_discriminator/*' --tag-name-filter cat -- --all
git rev-list --objects --all |   git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |   sed -n 's/^blob //p' |   sort --numeric-sort --key=2 |   cut -c 1-12,41- |   $(command -v gnumfmt || echo numfmt) --field=2 --to=iec-i --suffix=B --padding=7 --round=nearest
git gc --prune=now

```


## Log for 4/2/2020

Iterable dataset isn't really suited for multiprocessing.
The examples with subword mask beginning with 0 are filtered out for all punctuation tasks before anything rather than as a factor for the loss.

Parameters to tune: 
- Look at first subword token instead of last or all subword tokens. Not sure how to bring the labels to the first subwords. to look into this.
- Compare Electra to BERT (to Roberta)
- BiLSTM (nooooo)
- Dice (various gamma) vs CRF vs CEL (weighted)
- Immediate unfreeze 2 layers vs gradual unfreeze vs no unfreeze
- Optimizers

Experiments:
### CEL BERT novograd lr 0.00575 ted: blank and period overwhelm training on 1st epoch.
label            | precision    | recall   | f1     | support   
---|---|---|---|---
 (label_id: 0)   | 96.71        | 100.00   | 98.33  | 3702
! (label_id: 1)  | 0.00         | 0.00     | 0.00   | 115
, (label_id: 2)  | 0.00         | 0.00     | 0.00   | 12414
- (label_id: 3)  | 0.00         | 0.00     | 0.00   | 1164
. (label_id: 4)  | 40.31        | 99.56    | 57.38  | 10406
: (label_id: 5)  | 0.00         | 0.00     | 0.00   | 297
; (label_id: 6)  | 0.00         | 0.00     | 0.00   | 125
? (label_id: 7)  | 0.00         | 0.00     | 0.00   | 856
— (label_id: 8)  | 0.00         | 0.00     | 0.00   | 385
… (label_id: 9)  | 0.00         | 0.00     | 0.00   | 66

###Focal DistilBERT gamma 3 0 unfrozen ted
label               | precision   | recall | f1    | support   
---|---|---|---|---
 (label_id: 0)      | 100.00      | 51.29  | 67.80 | 4118
! (label_id: 1)     | 0.00        | 0.00   | 0.00  | 91
, (label_id: 2)     | 0.00        | 0.00   | 0.00  | 13953
\- (label_id: 3)    | 94.27       | 46.49  | 62.27 | 1310
. (label_id: 4)     | 39.51       | 99.94  | 56.63 | 12142
: (label_id: 5)     | 0.00        | 0.00   | 0.00  | 254
; (label_id: 6)     | 0.00        | 0.00   | 0.00  | 79
? (label_id: 7)     | 0.00        | 0.00   | 0.00  | 905
— (label_id: 8)     | 0.00        | 0.00   | 0.00  | 566
… (label_id: 9)     | 0.00        | 0.00   | 0.00  | 52

## Observations
- CRF tends to perform better on higher proportion classes like blank, comma and period without class weights.
- So far, the best performing model on F1 is dice with alpha 4, 1 unfrozen layer for 8 epoch.
- cel Doesnt quite converge, and weighted cel's blank class seems to suffer. To experiment further. with cel and focal
- Higher dice alpha results in better scores on weaker classes.
- Dice with class weights perform better than without



### elsmall dice alpha 4 weighted ted-l unfrozen 1 0.003162277660 lr adamw accgrad4 bbs8

label                 |   precision  |  recall |    f1    |      support
---|---|---|---|---
 (label_id: 0)        |      79.50   |   29.94 |   43.50  |    5026
! (label_id: 1)       |       6.84   |   20.59 |   10.27  |     102
, (label_id: 2)       |      50.70   |   60.09 |   55.00  |   17571
\- (label_id: 3)       |      64.45   |   82.11 |   72.22  |    1526
. (label_id: 4)       |      57.40   |   49.43 |   53.12  |   14767
: (label_id: 5)       |      17.86   |   31.83 |   22.89  |     289
; (label_id: 6)       |       1.50   |    5.88 |    2.39  |      85
? (label_id: 7)       |      37.02   |   61.32 |   46.17  |    1228
— (label_id: 8)       |       6.44   |    7.34 |    6.86  |     763
… (label_id: 9)       |       0.00   |    0.00 |    0.00  |      80
-------------------||||
micro avg             |      51.99   |   51.99 |   51.99  |   41437
macro avg             |      32.17   |   34.85 |   31.24  |   41437
weighted avg          |      55.33   |   51.99 |   51.87  |   41437

{'punct_f1': tensor(31.2411),
 'punct_precision': tensor(32.1728),
 'punct_recall': tensor(34.8539),
 'test_loss': tensor(0.6303)}


### elsmall dice alpha 1 weighted ted-l unfrozen 0.007943282347 lr adamw accgrad4 bbs7

label                 |   precision  |  recall |    f1    |      support
---|---|---|---|---
 (label_id: 0)             |     0.00  |  0.00 |   0.00  |  5026
! (label_id: 1)            |     0.00  |  0.00 |   0.00  |   102
, (label_id: 2)            |    42.79  | 47.54 |  45.04  | 17571
\- (label_id: 3)            |    73.63  | 80.87 |  77.08  |  1526
. (label_id: 4)            |    47.36  | 55.16 |  50.96  | 14767
: (label_id: 5)            |    10.88  | 27.68 |  15.62  |   289
; (label_id: 6)            |     0.00  |  0.00 |   0.00  |    85
? (label_id: 7)            |    43.18  | 60.10 |  50.26  |  1228
— (label_id: 8)            |     3.03  |  2.36 |   2.65  |   763
… (label_id: 9)            |     0.00  |  0.00 |   0.00  |    80
-------------------||||
micro avg                  |    44.81  | 44.81 |  44.81  | 41437
macro avg                  |    22.09  | 27.37 |  24.16  | 41437
weighted avg               |    39.14  | 44.81 |  41.75  | 41437

{'punct_f1': tensor(24.1611),
 'punct_precision': tensor(22.0869),
 'punct_recall': tensor(27.3705),
 'test_loss': tensor(0.4047)}


### elsmall crf ted-l unfrozen 0.005011872336272719 lr adamw accgrad4 bbs8

label                  |  precision | recall |   f1   |     support
---|---|---|---|---
 (label_id: 0)         |     59.35  |  52.35 |  55.63 |   7314
! (label_id: 1)        |      0.00  |   0.00 |   0.00 |    154
, (label_id: 2)        |     44.15  |  82.80 |  57.59 |  28180
\- (label_id: 3)        |      3.91  |   2.02 |   2.66 |   1933
. (label_id: 4)        |     39.91  |  11.64 |  18.02 |  24624
: (label_id: 5)        |      0.00  |   0.00 |   0.00 |    522
; (label_id: 6)        |      0.00  |   0.00 |   0.00 |    485
? (label_id: 7)        |      0.00  |   0.00 |   0.00 |   2096
— (label_id: 8)        |      0.00  |   0.00 |   0.00 |   2055
… (label_id: 9)        |      0.00  |   0.00 |   0.00 |    123
-------------------||||
micro avg              |     44.55  |  44.55 |  44.55 |  67486
macro avg              |     14.73  |  14.88 |  13.39 |  67486
weighted avg           |     39.54  |  44.55 |  36.73 |  67486

{'punct_f1': 13.390362739562988,
 'punct_precision': 14.73101806640625,
 'punct_recall': 14.881169319152832,
 'test_loss': 11.328206062316895}

### elsmall dice alpha 3 no weight ted-l unfrozen 0.005011872336272719 lr adamw accgrad4 bbs8
label                |  precision | recall |   f1   |    support
---|---|---|---|---
 (label_id: 0)       |  62.32   | 99.78 |  76.72  |   7314
! (label_id: 1)      |   0.00   |  0.00 |   0.00  |    154
, (label_id: 2)      |  49.81   |  4.72 |   8.62  |  28180
\- (label_id: 3)      |   5.91   | 28.35 |   9.78  |   1933
. (label_id: 4)      |  41.80   | 52.40 |  46.50  |  24624
: (label_id: 5)      |   0.94   |  4.02 |   1.53  |    522
; (label_id: 6)      |   0.00   |  0.00 |   0.00  |    485
? (label_id: 7)      |   4.92   | 24.86 |   8.22  |   2096
— (label_id: 8)      |   0.00   |  0.00 |   0.00  |   2055
… (label_id: 9)      |   0.00   |  0.00 |   0.00  |    123
-------------------||||
micro avg            |  33.52   | 33.52 |  33.52  |  67486
macro avg            |  16.57   | 21.41 |  15.14  |  67486
weighted avg         |  43.14   | 33.52 |  29.43  |  67486

'punct_f1': 15.136445999145508,
 'punct_precision': 16.57059097290039,
 'punct_recall': 21.41229820251465,
 'test_loss': 0.6608337163925171}


 ### elsmall dice alpha 5 weighted ted-l unfrozen 0 to 2 every 3 ep total 10 ep, 0.003981071705534973 lr adamw accgrad4 

 0 layer not too much improvement, 1 layer pretty decent.
 alpha 5 seems too high. to try full run 4 next.
layer 0 * 8 + layer 1 * 3
label                |  precision | recall |   f1   |    support
---|---|---|---|---
  (label_id: 0)        | 0.00     | 0.00    | 0.00  | 5704
! (label_id: 1)        | 0.00     | 0.00    | 0.00  | 110
, (label_id: 2)        | 0.00     | 0.00    | 0.00  | 19711
\- (label_id: 3)        | 6.82     | 29.32   | 11.07 | 1702
. (label_id: 4)        | 37.30    | 83.82   | 51.62 | 18406
: (label_id: 5)        | 0.00     | 0.00    | 0.00  | 379
; (label_id: 6)        | 0.00     | 0.00    | 0.00  | 190
? (label_id: 7)        | 6.71     | 1.31    | 2.20  | 1446
— (label_id: 8)        | 0.00     | 0.00    | 0.00  | 1227
… (label_id: 9)        | 0.00     | 0.00    | 0.00  | 86
-------------------||||
micro avg              | 32.57    | 32.57   | 32.57 | 48961
macro avg              | 5.08     | 11.44   | 6.49  | 48961
weighted avg           | 14.46    | 32.57   | 19.86 | 48961

 {'punct_f1': 6.104840278625488,
 'punct_precision': 4.423948764801025,
 'punct_recall': 11.572192192077637,
 'test_loss': 0.47498953342437744}

 ### elsmall dice alpha 4 weighted ted-l unfrozen 0 to 2 every 3 ep total 10 ep, 0.008413951416451957 lr adamw accgrad4 
try early_stop_threshold=None for lr_finder
lr 0 : 0.008413951416451957
1: 0.00031622776601683794 ** too high. to adjust the min to 1e-10?
2: 0.00031622776601683794

label                |  precision | recall |   f1   |    support
---|---|---|---|---
 (label_id: 0)       |   0.00  | 0.00     | 0.00  | 7470
! (label_id: 1)      |   0.00  | 0.00     | 0.00  | 148
, (label_id: 2)      |   0.00  | 0.00     | 0.00  | 28513
\- (label_id: 3)      |   3.02  | 100.00   | 5.86  | 2074
. (label_id: 4)      |   0.00  | 0.00     | 0.00  | 25120
: (label_id: 5)      |   0.00  | 0.00     | 0.00  | 570
; (label_id: 6)      |   0.00  | 0.00     | 0.00  | 534
? (label_id: 7)      |   0.00  | 0.00     | 0.00  | 2085
— (label_id: 8)      |   0.00  | 0.00     | 0.00  | 2073
… (label_id: 9)      |   0.00  | 0.00     | 0.00  | 142

 {'punct_f1': 0.5858508944511414,
 'punct_precision': 0.30176490545272827,
 'punct_recall': 10.0,
 'test_loss': 0.8140875697135925}

### elsmall dice alpha 3 unweighted ted-l unfrozen 0-2 2 ep 

label                |  precision | recall |   f1   |    support
---|---|---|---|---
 (label_id: 0)      | 62.15  | 100.00   | 76.66   | 5154
! (label_id: 1)     | 0.00   | 0.00     | 0.00    | 108
, (label_id: 2)     | 0.00   | 0.00     | 0.00    | 18022
\- (label_id: 3)     | 0.00   | 0.00     | 0.00    | 1557
. (label_id: 4)     | 41.74  | 94.01    | 57.81   | 15164
: (label_id: 5)     | 0.00   | 0.00     | 0.00    | 319
; (label_id: 6)     | 0.00   | 0.00     | 0.00    | 88
? (label_id: 7)     | 0.00   | 0.00     | 0.00    | 1217
 (label_id: 8)      | 0.00   | 0.00     | 0.00    | 752
… (label_id: 9)     | 0.00   | 0.00     | 0.00    | 67
-------------------||||
micro avg           | 45.72 | 45.72 | 45.72 | 42448
macro avg           | 10.39 | 19.40 | 13.45 | 42448
weighted avg        | 22.46 | 45.72 | 29.96 | 42448

{ 'punct_f1': 13.446383476257324,
 'punct_precision': 10.388500213623047,
 'punct_recall': 19.400554656982422,
 'test_loss': 0.44148480892181396}


 ## Log for 8/2/2021

 I believe Dice loss performs better when unweighted, and the experiments that failed to converge were due to it being weighted.
 For 3 layers unfreezing, the tuning of the 1st and 2nd layers result in some form of divergence, I believe the training process causes divergence and requires a much smaller learning rate.

For /2021-02-08_07-56-46 crf adam, frozen best lr was 0.01, auto set to 0.007943282347242822.
End frozen 
micro avg      |  41.42 |   41.42  |    41.42   |   33406
macro avg      |  11.01 |   13.54  |    11.07   |   33406
weighted avg   |  34.88 |   41.42  |    34.20   |   33406

1st layer best lr 1e-10, set to 0.007943282347242822
micro avg            |       36.65  |    36.65  |    36.65  |    33463
macro avg            |       10.71  |     9.91  |     8.26  |    33463
weighted avg         |       34.32  |    36.65  |    31.08  |    33463

2nd layer best lr 1e-10, set to 0.007943282347242822
micro avg        |   35.72  |    35.72  |    35.72  |    42448
macro avg        |    3.57  |    10.00  |     5.26  |    42448
weighted avg     |   12.76  |    35.72  |    18.81  |    42448

{'punct_f1': 5.264181137084961,
 'punct_precision': 3.572371006011963,
 'punct_recall': 10.0,
 'test_loss': 18.49854850769043}


For 2021-02-08_08-37-54/ dice adamw, the frozen best lr was at 0.01, auto set to 0.005011872336272725.
alpha from 3->4 seems to reduce convergence rate.

micro avg        |   50.98  | 50.98  |  50.98    |  33463
macro avg        |   25.99  | 25.38  |  23.38    |  33463
weighted avg     |   50.31  | 50.98  |  48.27    |  33463

unfreeze 1 0.0025118864315095825 best lr 1e-10, 
micro avg     |  58.55  |  58.55 |  58.55 | 39340
macro avg     |  30.02  |  29.74 |  29.52 | 39340
weighted avg  |  57.91  |  58.55 |  57.51 | 39340

still increasing?!
{'punct_f1': 29.523975372314453,
 'punct_precision': 30.015613555908203,
 'punct_recall': 29.738296508789062,
 'test_loss': 0.3690211772918701}


### Implemented mlp 2 layer before classifier, 

adamw mean 2 layer domain, dice, alpha 4 10 batch, accgrad 4 2021-02-08_11-07-07/
frozen lr 0.0025118864315095825 best: 0.01,

unfreeze 0.07943282347242822 best lr 1e-10

ep 6 
micro avg    | 64.21 | 64.21 | 64.21 | 33835
macro avg    | 36.55 | 37.56 | 36.71 | 33835
weighted avg | 63.77 | 64.21 | 63.91 | 33835

{'punct_f1': 38.96394729614258,
 'punct_precision': 38.412635803222656,
 'punct_recall': 40.2258415222168,
 'test_loss': 0.2748030722141266}


 ### CEL

  (label_id: 0)                                         100.00     100.00     100.00       5564
! (label_id: 1)                                          0.00       0.00       0.00        148
, (label_id: 2)                                         69.27      76.77      72.83      19606
\- (label_id: 3)                                         87.16      75.17      80.72       1788
. (label_id: 4)                                         65.71      68.86      67.25      16090
: (label_id: 5)                                          0.00       0.00       0.00        368
; (label_id: 6)                                          0.00       0.00       0.00        202
? (label_id: 7)                                         47.76      17.08      25.16       1370
 (label_id: 8)                                          0.00       0.00       0.00        934
… (label_id: 9)                                          0.00       0.00       0.00        122
-------------------
micro avg                                               72.03      72.03      72.03      46192
macro avg                                               36.99      33.79      34.60      46192
weighted avg                                            69.12      72.03      70.25      46192

{'punct_f1': 34.595890045166016,
 'punct_precision': 36.98928451538086,
 'punct_recall': 33.78831481933594,
 'test_loss': 0.2638570964336395}



### domain adversarial dice 3, open l ted ul 
initial_lr 0.007943282347242822

* Half of samples are thrown away, so would have to adjust either non-adversarial half the samples or the adversarial train twice as long. To see if both can converge.
testing gamma 0.1 vs just open subtitles:

https://www.aclweb.org/anthology/2020.acl-main.370.pdf uses the formula of 2/(1+e^(-10p))-1 where p varies from 0 to 1. To repeat cycle every unfrozen layer.

### 2021-02-09_16-21-19 warmup ted 

: (label_id: 5)                                         20.69      19.57      20.11        368
; (label_id: 6)                                          0.00       0.00       0.00        200
? (label_id: 7)                                         22.42      29.15      25.35       1372
 (label_id: 8)                                          6.83       9.44       7.93        932
… (label_id: 9)                                          0.00       0.00       0.00        124
-------------------
micro avg                                               89.82      89.82      89.82     300124
macro avg                                               31.58      32.56      31.95     300124
weighted avg                                            90.46      89.82      90.11     300124

[INFO] - Domain report:
label                                                precision    recall       f1           support
0 (label_id: 0)                                        100.00     100.00     100.00       2744
-------------------
micro avg                                              100.00     100.00     100.00       2744
macro avg                                              100.00     100.00     100.00       2744
weighted avg                                           100.00     100.00     100.00       2744

Testing: 100%|| 100/100 [00:10<00:00,  9.74it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'domain_f1': 100.0,
 'domain_precision': 100.0,
 'domain_recall': 100.0,
 'punct_f1': 31.946725845336914,
 'punct_precision': 31.575754165649414,
 'punct_recall': 32.5594596862793,
 'test_loss': 0.23392203450202942}

### 2021-02-09_16-44-40 cosine ted around the same:

 (label_id: 0)                                          97.17      95.67      96.41     259964
! (label_id: 1)                                          0.00       0.00       0.00        152
, (label_id: 2)                                         43.51      47.93      45.61      19336
- (label_id: 3)                                         69.47      61.49      65.23       1776
. (label_id: 4)                                         55.49      62.29      58.69      15900
: (label_id: 5)                                         20.45      19.57      20.00        368
; (label_id: 6)                                          0.00       0.00       0.00        200
? (label_id: 7)                                         22.67      29.74      25.73       1372
 (label_id: 8)                                          6.85       9.44       7.94        932
… (label_id: 9)                                          0.00       0.00       0.00        124
-------------------
micro avg                                               89.81      89.81      89.81     300124
macro avg                                               31.56      32.61      31.96     300124
weighted avg                                            90.47      89.81      90.11     300124

[INFO] - Domain report:
label                                                precision    recall       f1           support
0 (label_id: 0)                                        100.00     100.00     100.00       2744
-------------------
micro avg                                              100.00     100.00     100.00       2744
macro avg                                              100.00     100.00     100.00       2744
weighted avg                                           100.00     100.00     100.00       2744

Testing: 100%|| 100/100 [00:10<00:00,  9.29it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'domain_f1': 100.0,
 'domain_precision': 100.0,
 'domain_recall': 100.0,
 'punct_f1': 31.962158203125,
 'punct_precision': 31.560827255249023,
 'punct_recall': 32.61237335205078,
 'test_loss': 0.23370929062366486}

 #####################################################################
### 2021-02-09_16-54-29 domain adversarial

### 2021-02-09_17-19-42 just open subtitles to compare train loss