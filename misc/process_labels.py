# Let's see if I can gleam some meaningful information
# from the labels of these files.

import os
import sys

import pandas as pd
import numpy as np

keyword="snare"

stopwords = ['snare','Snare','SP','KSHMR','Cymatics','.wav','LUN','ASC','BFBS2',\
                 '1','2','3','4','5','6','7','8','9','0',\
                 'SS FBS', ' ','FBW','-','SampleMagic',
                 'Ultimate','Posty','Congratulate','EK','BFB','SNARE',
                 'A#','C#','D#','F#','G#','()','Titan','JU','ZAY']
def clean(s) :
    for sw in stopwords :
        s = s.replace(sw,'')
    return s

folders = os.listdir("clusters/"+keyword+"/")
gfold   = []
s_kw    = []

for fo in folders :
    try :
        files = os.listdir("clusters/"+keyword+"/"+fo)
    except :
        continue
    print("Results for folder: "+str(fo))
    gfold.append(fo)
    wt = []
    for f in files :
        words = clean(f).split("_")
        words = filter(None,words)
        if len(words)>0 :
            wt.extend(words)
        #    print(words)
    wt = list(set(wt))
    s_kw.append(wt)

df = pd.DataFrame(s_kw,index=gfold)
print(df)
df.to_csv(keyword+"_labels.csv")
