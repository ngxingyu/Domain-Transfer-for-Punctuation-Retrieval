# ASR
##Datasources
The chosen datasources for this project are:
1. [TED - Ultimate Dataset | Kaggle](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset)-A collection of 4005 TED talks.
2. [Untokenised Corpus files for Opensubtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php) - Select the rightmost column language ID, in my case en.
3. BookCorpusOpen from Huggingface Datasets, a precompiled collection of 17868 books.


##Preprocessing
The following steps are taken for preprocessing the data to a useable form.

1. (Important in the case of TED dataset) Remove speaker tags i.e. Narrator: ... which will not be spoken.
2. Remove tags that are added for better readability, i.e. sound effects (Applause) etc.
3. Identify spoken text that are within square or round brackets and remove the brackets.
4. Remove music lyrics i.e. ♫Lyrics♫ as a large portion of these do not contain appropriate punctuation information.
5. Remove empty matching tags- square brackets, parentheses, single/double quotes. For single/double quotes, it may be used when training for retrieval of the quotes but I'll look into it again when I'm working on that.
6. Convert ellipsis to the unicode version.
7. Removing Non-sentence punctuation (All punctuation that are non-readable (@$#%&^+•=€²£¥) and not (.?!,;:-–—)).
8. Combining repeated patterns of punctuation, i.e. [. .] to [.], [!!!!] to [!].
9. Remove excess whitespaces
