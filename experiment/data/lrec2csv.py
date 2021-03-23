import pandas as pd
fntg={"dev2012":'lrec.dev',"test2011":'lrec.test',"test2011asr":'lrec.testasr',"train2012":'lrec.train'}
for fn,tg in fntg.items():
    f=pd.read_csv(f'/home/nxingyu2/data/punct-restoration/punctuation-restoration/data/en/{fn}',sep='\t',header=None)
    m={'O':' ','PERIOD':". ",'COMMA':", ","QUESTION":'?'}
    s=''.join([str(w)+str(m[p])for _,w,p in f.itertuples()])
    df=pd.DataFrame([[fn,s]],columns=['id','transcript'])
    df.to_csv(f'~/data/{tg}.csv',index=None)
