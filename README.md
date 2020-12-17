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
- 30341     0.0367  | 3966  0.03693     | 3757  0.03631
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
- 10479724  0.09805945244798349     | 1321067 0.09882862227992877   | 1324557 0.09893127451608667
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
