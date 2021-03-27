import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Enter dff file location.')
    parser.add_argument("-i", "--input", dest="filename", required=True,
                        help="input directory", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output", required=True,
                        help="output directory", metavar="FILE")

    args = parser.parse_args()
    f=open(args.filename,'r')
    fntg={"dev2012":'lrec.dev',"test2011":'lrec.test',"test2011asr":'lrec.testasr',"train2012":'lrec.train'}
    for fn,tg in fntg.items():
        f=pd.read_csv(f'{args.input}/{fn}',sep='\t',header=None)
        m={'O':' ','PERIOD':". ",'COMMA':", ","QUESTION":'?'}
        s=''.join([str(w)+str(m[p])for _,w,p in f.itertuples()])
        df=pd.DataFrame([[fn,s]],columns=['id','transcript'])
        df.to_csv(f'{args.output}/{tg}.csv',index=None)