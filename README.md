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
. 327534 0.3945194589320774 | 42447 0.40324327405380755 | 38920  0.39807745236954506
? 29405 0.035418749473024896 | 3825 0.036337209302325583 | 3544 0.03618837560756443
! 2698 0.003249780176100023 | 278 0.0026409788721690228 | 301 0.003073561246579259
, 397931 0.47931366762626326 | 49507 0.47031273749810004 | 47137 0.4813237756810848
; 4770 0.005745534262415534 | 602 0.005718954248366013 | 474 0.004840093125842421
: 10120  0.012189686946676142 | 1295 0.012302401580787353 | 1178 0.012028754646080954
\- 30778 0.03707254790956505 | 3813 0.036223210214318284 | 3474 0.03547359392231344
— 26974 0.03249057467387769 | 3497 0.033221234230126157 | 2904 0.029653228770983947

For subtitles corpus
Train | Val | Test
--- | --- | ---
. 51669337 0.4692522875161914 | 6439837 0.46993458987098946 | 6551814 0.4683982355926993
? 13574073 0.12327746350145292 | 1687622 0.12315093571891012 | 1713797 0.12252171550719561
! 6695030 0.06080314408697613 | 823378 0.0600844093940259 | 845345 0.06043488207496586
, 26494742 0.2406208210229466 | 3299631 0.24078415970941544 | 3381518 0.24174939411053997
; 62177 0.0005646811276268986 | 7932 0.0005788222849206724 | 6795 0.00048578393874618413
: 437205 0.00397062277697731 | 46257 0.0033755146789681725 | 52558 0.0037574440401209634
\- 11166241 0.10140993549437426 | 1398018 0.10201764663643831 | 1435039 0.10259292092338268
— 11126 0.00010104447345444254 | 1013 0.0000739217063319013 | 834 0.00005962381234942128

## Processing part-2
The punctuation to be classified are as follows: , . ! ? - hyphen — emdash : :
There are occurences of consecutive punctuation. This includes: 
1. ., : period after abbreviation or initial
2. ?, or !— etc. where the first punctuation applies to a local scope and the 2nd applies to a larger context.
In most cases, it makes more sense to classify the punctuations from right to left, so I'll append punctuations to the previous word and predict the punctuation at the top of the stack.:
