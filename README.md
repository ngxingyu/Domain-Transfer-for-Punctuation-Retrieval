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
. 325322 0.39519049371783144 | 42673 0.3981581697394939 | 40906 0.39620320596639064
? 29249 0.035530725714070524 | 3779 0.03525975964768232 | 3746 0.03628262869872633
! 2618 0.0031802605189728416 | 355 0.003312308725834142 | 304 0.002944452515860332
, 394500 0.47922565879861956 | 50863 0.47457453161155483 | 49212 0.4766526224030219
; 4633 0.0056280164187934205 | 651 0.0060741210718817645 | 562 0.0054433628747154825
: 10138 0.012315309832447162 | 1366 0.012745390759125176 | 1308 0.012668894377451693
- 30341 0.03685725149203781 | 3966 0.03700455325819213 | 3757 0.03638917138844496
— 26402 0.03207228350722726 | 3523 0.03287116518623572 | 3450 0.03341566177538864
words 5842593 | 757511 | 733686

For subtitles corpus
Train | Val | Test
--- | --- | ---
. 47443035 0.4620226396087783 | 5921757 0.4610514647042913 | 5921304 0.46015088848235636
? 13250829 0.12904281927968034 | 1660096 0.12925043907572284 | 1670946 0.12985100689071857
! 6519047 0.06348555278290455 | 813038 0.06330086843486614 | 834767 0.06487063942769214
, 24551760 0.23909661264801502 | 3062385 0.23842874500565478 | 3063118 0.23803818706593993
; 57285 0.0005578683343084789 | 8073 0.0006285412377707737 | 7344 0.0005707101214554134
: 374276 0.003644876122748368 | 57019 0.004439340125907562 | 44933 0.0034917916513284436
- 10479724 0.1020564924830687 | 1321067 0.10285458780603528 | 1324557 0.10293274596195778
— 9564 9.313874049622575e-05 | 591 4.6013609751334976e-05 | 1210 9.403039855134125e-05

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
