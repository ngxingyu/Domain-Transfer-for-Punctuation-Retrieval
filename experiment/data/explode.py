#!/usr/bin/env python
#%%
import pandas as pd
import regex as re
import argparse, os, csv

from tqdm import tqdm
import subprocess

def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

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

    parser = argparse.ArgumentParser(description='Enter csv file location. Extracts the "talk_id" and "transcript" column from csv and explodes the transcript by char length')
    parser.add_argument("-i", "--input", dest="filename", required=True, type=validate_file,
                        help="input file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output", required=True,
                        help="output csv filepath", metavar="FILE")
    parser.add_argument("-l","--length", type=int, required=False, default=300,help="split charlen")
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
    open(args.output, 'w').close()
    with open(args.output, 'a') as f:
      for i in tqdm(o,total=total):
          i = i.iloc[:,[0,-1]]
          i.columns=['id','transcript']
          i.transcript=i.transcript.str.findall(f'\\w.{{{args.length},}}?[.!?… ]*[.!?…]|\\w.+?[.!?… ]*[.!?…]')
          i=i.explode('transcript')
          i=i.loc[~i.transcript.isnull()]
          i.to_csv(f, mode='a', index=False, header=False)




