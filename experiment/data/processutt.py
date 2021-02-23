#!/usr/bin/env python
#%%
import pandas as pd
import regex as re
import argparse, os, csv
import unicodedata
#%%
def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f
#%%
tags=list('.?!,;:-—…')
parentheses=r'\([^)(]+[^)( ] *\)'
parenthesestokeep=r'\([^)(]+[^)(.!?—\-, ] *\)'
speakertag=r'((?<=[^\w\d \",])|^) *(?![?\.,!:\-\—\[\]\(\)])(?:[A-Z\d][^\s.?!\[\]\(\)]*\s?)*:(?=[^\w]*[A-Z])'#lookahead keeps semicolon in false cases.
parenthesestoremove=r'\(([^\w]*[^\(\)]+[\w ]+)\):?'
parenthesesaroundsentence=r'\(([^\w]*[^\(\)]+\W*)\):?'
squarebracketsaroundsentence=r'\[([^\[\]]+)\]' #generic since it seems like the square brackets just denote unclear speech.

def displayinstances(col,exp):
    for i in range(len(col)):
        #     temp={x.group() for x in re.finditer( , tedtalks[i])}
        temp={x.group() for x in re.finditer(exp, col[i])}
        if len(temp)!=0:print(i,temp)
    print('--fin--')

''' Identifies term to remove if the words from the previous
    punctuation (except ") through : until the next word all
    begins with a caps. Drawback:This doesnt properly capture
    places where the following term is caps due to it being a
    proper noun, where the prefix will be removed regardless
    but will not break the syntax.
'''

def remove_disf(text):
    s=re.sub("[\{\}\[\]/\(\) ]+[A-Z]? +"," ",text)
    s=re.sub("\<+([^<>]+)\>+",'',s)
    s=re.sub("[A-Z]+\.[0-9]+:","",s)
    s=re.sub(" *[#\+] *"," ",s)
    s=re.sub("@@",' ',s)
    return s

def to_emdash(s):
    return re.sub('--','…',s)

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFKD', s)
                  if unicodedata.category(c) != 'Mn')

def removespeakertags(text):
    return re.sub(speakertag,' ',text)

def removenametags(text):
    # return re.sub(r"(?<=[a-z][.?!;]) *[ A-z.,\-']{1,25}:",' ',text)
    return re.sub(r"(?<=[a-z][.?!;])([\(\[]* *)[ A-Za-z.,\-']{1,25}:", "\g<1>",text)

def removeparentheses(text):
    return re.sub(parenthesestoremove, ' ',text)

def removeparenthesesaroundsentence(text):
    return re.sub(parenthesesaroundsentence,r'\g<1>',text)

def removedashafterpunct(text):
    return re.sub(r"([^A-Za-zÀ-ÖØ-öø-ÿ0-9 ]+ *)-+( *[^- ])",r"\g<1> \g<2>",text)

def removesquarebrackets(text):
    return re.sub(squarebracketsaroundsentence, r'\g<1>',text)

def removemusic(text):
    text = re.sub(r'♫( *[^♫ ])+ *♫', ' ',text)
    return re.sub(r'♪( *[^♫ ])+ *♪', ' ',text)

def reducewhitespaces(text):
    text=re.sub(r'(?<=[.?!,;:\—\-]) *(?=[.?!,;:\—\-])','',text)
    return re.sub(r'\s+', ' ',text)

def removeemptyquotes(text):
    text= re.sub(r"'[_^\W]*'",' ',text)
    text= re.sub(r"\([_^\W]*\)",' ',text)
    text= re.sub(r"\[[_^\W]*\]",' ',text)
    return re.sub(r'"[_^\W]*"',' ',text)

def ellipsistounicode(text):
    text = re.sub(r'\.{3,}(?= )','…',text) #ellipsis without trailing punctuation
    return re.sub(r'\.{3,}([^\w\s])','…\g<1>',text) #ellipsis with trailing punctuation

def removenonsentencepunct(text):
    return re.sub(r'[^A-Za-z\d\s$%&+=€²£¢¥…,.!?;:\-\–\—\']',' ',text)

def combinerepeatedpunct(text):
    newtext=[text,re.sub(r'([_^\W]+) *\1+','\g<1> ',text)]
    i=1
    while (newtext[0]!=newtext[1]):
        i+=1
        newtext[i%2]=re.sub(r'([_^\W]+) *\1+','\g<1> ',newtext[(1+i)%2])
    return newtext[i%2]

def endashtohyphen(text):
    return re.sub('–','-',text)

def removedashafterpunct(text):
    return re.sub(r"([^A-Za-z0-9 ]+ *)-+( *[^- ])",r"\g<1> \g<2>",text)

def pronouncesymbol(text):
    text=re.sub("\$ *([\d](\.[\d])?+)", "\g<1> dollars ",text)
    text=re.sub('\£ *([\d](\.[\d])?+)', " pounds ",text)
    text=re.sub("\$", " dollars ",text)
    text=re.sub("\£", " pounds ",text)
    text=re.sub('€', " euro ",text)
    text=re.sub('¥', " yen ",text)
    text=re.sub("¢"," cents ",text)
    text=re.sub('(?<=\d)\.(?=\d)',' point ',text)
    text=re.sub('\+',' plus ',text)
    text=re.sub('%',' percent ',text)
    text=re.sub('²',' squared ',text)
    text=re.sub('&', ' and ',text)
    return text

def stripleadingpunctuation(text):
    return re.sub(r'^[^A-Z]*','',text)

def striptrailingtext(text):
    return re.sub(r'[^!.?…;]*$','',text)

def preprocess(tedtalks):
    tedtalks=remove_disf(tedtalks)
    tedtalks=strip_accents(tedtalks)
    tedtalks=removespeakertags(tedtalks)
    tedtalks=removenametags(tedtalks) # Remove *Mr Brown: *Hi!
    tedtalks=removemusic(tedtalks)
    tedtalks=removeemptyquotes(tedtalks)
    tedtalks=removenonsentencepunct(tedtalks)
    tedtalks=ellipsistounicode(tedtalks)
    tedtalks=to_emdash(tedtalks)
    tedtalks=endashtohyphen(tedtalks)
    tedtalks=removedashafterpunct(tedtalks)
    tedtalks=combinerepeatedpunct(tedtalks)
    tedtalks=pronouncesymbol(tedtalks)
    tedtalks=stripleadingpunctuation(tedtalks)
    tedtalks=striptrailingtext(tedtalks)
    tedtalks=reducewhitespaces(tedtalks)
    return tedtalks


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Enter dff file location.')
    parser.add_argument("-i", "--input", dest="filename", required=True, type=validate_file,
                        help="input file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output", required=True,
                        help="output file", metavar="FILE")

    args = parser.parse_args()
    f=open(args.filename,'r')
    started=False
    script=""
    for line in f:
        if started==False and re.match('^=+',line):
            started=True
            continue
        if started:
            line=re.sub('.*utt[0-9]+:','',line)
            script+=re.sub('\n',' ',line)
    with open (args.output, 'a') as f:
        writer=csv.writer(f)
        script=preprocess(script).strip()
        if bool(script):
            writer.writerow((args.filename,script))
