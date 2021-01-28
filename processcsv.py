#!/usr/bin/env python
#%%
import pandas as pd
import regex as re
import argparse, os, csv
import unicodedata

from tqdm import tqdm
import subprocess

# import dask.dataframe as dd

def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


parentheses=r'\([^)(]+[^)( ] *\)'
parenthesestokeep=r'\([^)(]+[^)(.!?—\-, ] *\)'
speakertag=r'((?<=[^\w\d \",])|^) *(?![?\.,!:\-\—\[\]\(\)])(?:[A-Z\d][^\s.?!\[\]\(\)]*\s?)*:(?=[^\w]*[A-Z])'#lookahead keeps semicolon in false cases.
parenthesestoremove=r'\(([^\(\)]+[\w ]+)\):?'
parenthesesaroundsentence=r'\(([^\w]*[^\(\)]+[_^\W]+)\):?'
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
    but will not break the syntax. '''

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
    print('stripping accents')
    tedtalks=tedtalks.apply(strip_accents)
    print('removing speaker tags')
    tedtalks=tedtalks.apply(removespeakertags)
    print('removing name tags')
    tedtalks=tedtalks.apply(removenametags) # Remove *Mr Brown: *Hi!
    print('removing non-sentence parenthesis')
    tedtalks=tedtalks.apply(removeparentheses) # Remove (Whispers) without punct
    print('removing parenthesis')
    tedtalks=tedtalks.apply(removeparenthesesaroundsentence) #Remove -> (<- Hi Everyone! ->)<-
    print('removing square brackets')
    tedtalks=tedtalks.apply(removesquarebrackets) #Remove entire [unclear text]
    print('removing music lyrics')
    tedtalks=tedtalks.apply(removemusic)
    print('removing empty tags')
    tedtalks=tedtalks.apply(removeemptyquotes)
    print('removing non-sentence punctuation')
    tedtalks=tedtalks.apply(removenonsentencepunct)
    print('change to unicode ellipsis')
    tedtalks=tedtalks.apply(ellipsistounicode)
    print('endash to hyphen')
    tedtalks=tedtalks.apply(endashtohyphen)
    print('remove hyphen after punct')
    tedtalks=tedtalks.apply(removedashafterpunct)
    print('combine repeated punctuation')
    tedtalks=tedtalks.apply(combinerepeatedpunct)
    print('pronounce symbol')
    tedtalks=tedtalks.apply(pronouncesymbol)
    print('strip leading')
    tedtalks=tedtalks.apply(stripleadingpunctuation)
    print('strip trailing')
    tedtalks=tedtalks.apply(striptrailingtext)
    print('reduce whitespaces')
    tedtalks=tedtalks.apply(reducewhitespaces)
    
    print('tolower')
    tedtalks=tedtalks.str.lower()

    print('--done--')
    return tedtalks

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Enter csv file location. Extracts the "talk_id" and "transcript" column from csv and preprocesses transcript to the format for sentence punctuation prediction')
    parser.add_argument("-i", "--input", dest="filename", required=True, type=validate_file,
                        help="input file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output", required=True,
                        help="output csv filepath", metavar="FILE")
    # parser.add_argument("-p","--preprocess", type=str2bool, nargs='?',
    #                     const=True, default=True,
    #                     help="requires preprocess?")
    parser.add_argument('-c',"--chunksize", dest='chunksize', type=int, required=False, default=2000)
    parser.add_argument('-s',"--header", dest='header', type=int, required=False, default=1)
    args = parser.parse_args()
    
    
    nb_samples=int(subprocess.Popen(['wc', '-l', args.filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])
    total = int((nb_samples+args.chunksize) / args.chunksize)
    print(total,nb_samples,args.chunksize)
    o=pd.read_csv(args.filename,
                  dtype='str',
                  # columns=['talk_id','transcript']
                  skiprows=range(0,args.header),
                  header=None,
                  chunksize=args.chunksize)
    # paths=os.path.splitext(args.output)
    open(args.output, 'w').close()
    with open(args.output, 'a') as f:
      for i in tqdm(o,total=total):
          i = i.iloc[:,[0,-1]]
          i.columns=['id','transcript']
        #   print(i)
          i=i.loc[~i.transcript.isnull()]
          i.loc[:,'transcript'] = preprocess(i.transcript.astype(str))
        #   print(1,i.transcript)
          i=i.loc[i.transcript.map(lambda x:len(x.split())>=1)]
        #   print(2)
          i.to_csv(f, mode='a', index=False, header=False)



#%%

'''
import pandas as pd

filename='../data/ted_talks_en.csv'
chunksize=4
header=1
total=0
o=pd.read_csv(filename,
                dtype='str',
                # columns=['talk_id','transcript'],
                skiprows=range(0,header),
                header=None,
                chunksize=chunksize)

i=next(iter(o)).iloc[:,[0,-1]]
i.columns=['talk_id','transcript']
i
# paths=os.path.splitext(args.output)
# open(args.output, 'w').close()
# with open(args.output, 'a') as f:
#     for i in tqdm(o,total=total):
#         i = i.iloc[:,[0,-1]]
#         print(i)
#         i.dropna(inplace=True)
#         i.iloc[:,1] = preprocess(i.iloc[:,1].astype(str))
#         i=i[i.iloc[:,-1].map(lambda x:len(x.split())>=1)]
#         i.to_csv(f, mode='a', index=False, header=False)

#%%'''