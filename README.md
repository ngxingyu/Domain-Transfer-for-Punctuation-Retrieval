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
9. Remove excess whitespaces
10. Remove examples with length < 10 words
11. Random shuffle with seed 42
12. Perform train dev test split of 0.9 0.1 0.1.

### Punctuation proportion
For TED corpus
Train | Val | Test
--- | --- | ---
. 328524 0.3951103702581199 | 42582 0.40382755154296984 | 39051 0.39807745236954506
? 29405 0.03536490617866584 | 3825 0.03627449120877037 | 3544 0.03612676989571759
! 2698 0.0032448398867553287 | 278 0.002636420537526317 | 301 0.0030683289330166465
, 397931 0.47858501889415667 | 49507 0.46950097680329267 | 47137 0.48050438842393906
; 4770 0.005736799948044076 | 602 0.0057090833222692185 | 474 0.004831853535713922
: 10394 0.012500691542970677 | 1342 0.01272689338618819 | 1214 0.012375253570372786
\- 30778 0.03701619052429781 | 3813 0.036160688883409516 | 3474 0.03541320502757418
— 26974 0.03244118276698971 | 3497 0.03316389431557385 | 2904 0.029602748244120736

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
