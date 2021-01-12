#!/usr/bin/env python
#%%
import pandas as pd
import regex as re
import argparse, os, csv
import xml.etree.ElementTree as ET
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

def removespeakertags(text):
    return re.sub(speakertag,' ',text)

def removeparentheses(text):
    return re.sub(parenthesestoremove, ' ',text)

def removeparenthesesaroundsentence(text):
    return re.sub(parenthesesaroundsentence,r'\g<1>',text)

def removesquarebrackets(text):
    return re.sub(squarebracketsaroundsentence, r'\g<1>',text)

def removemusic(text):
    text = re.sub(r'♫( *[^♫ ])+ *♫', ' ',text)
    return re.sub(r'♪( *[^♫ ])+ *♪', ' ',text)

def reducewhitespaces(text):
    text=re.sub(r'(?<=[.?!,;:\—\-]) *(?=[.?!,;:\—\-])','',text)
    return re.sub(r'\s+', ' ',text)

def removeemptyquotes(text):
    text= re.sub(r"'[^\w\d]*'",' ',text)
    text= re.sub(r"\([^\w\d]*\)",' ',text)
    text= re.sub(r"\[[^\w\d]*\]",' ',text)
    return re.sub(r'"[^\w\d]*"',' ',text)

def ellipsistounicode(text):
    text = re.sub(r'\.{3,}(?= )','…',text) #ellipsis without trailing punctuation
    return re.sub(r'\.{3,}([^\w\s])','…\g<1>',text) #ellipsis with trailing punctuation

def removenonsentencepunct(text):
    return re.sub(r'[^\w\d\s,.!?;:$#%&^+•=€²£¥…@\-\–\—\/](?!\w)|(?<!\w)[^\w\d\s,.!?;:$#%&^+•=€²£¥…@\-\–\—\/]',' ',text)

def combinerepeatedpunct(text):
    newtext=[text,re.sub(r'([^\w\d]+) *\1+','\g<1> ',text)]
    i=1
    while (newtext[0]!=newtext[1]):
        i+=1
        newtext[i%2]=re.sub(r'([^\w\d]+) *\1+','\g<1> ',newtext[(1+i)%2])
    return newtext[i%2]

def endashtohyphen(text):
    return re.sub('–','-',text)

def pronouncesymbol(text):
    return re.sub('(?<=\d)\.(?=\d)',' point ',text)


def preprocess(tedtalks):
    tedtalks=removespeakertags(tedtalks)
    tedtalks=removeparentheses(tedtalks)
    tedtalks=removeparenthesesaroundsentence(tedtalks)
    tedtalks=removesquarebrackets(tedtalks)
    tedtalks=removemusic(tedtalks)
    tedtalks=removeemptyquotes(tedtalks)
    tedtalks=ellipsistounicode(tedtalks)
    tedtalks=removenonsentencepunct(tedtalks)
    tedtalks=endashtohyphen(tedtalks)
    tedtalks=combinerepeatedpunct(tedtalks)
    tedtalks=pronouncesymbol(tedtalks)
    tedtalks=reducewhitespaces(tedtalks)
    return tedtalks


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Enter xml file location.')
    parser.add_argument("-i", "--input", dest="filename", required=True, type=validate_file,
                        help="input file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output", required=True,
                        help="output file", metavar="FILE")

    args = parser.parse_args()
    tree=ET.parse(args.filename)
    rows=[]
    for child in tree.getroot():
        rows.append(''.join([x.strip() for x in list(child.itertext())]))
    with open (args.output, 'a') as f:
        writer=csv.writer(f)
        writer.writerow((args.filename,preprocess(' '.join([i for i in rows if i.strip()[-1] in tags]))))
