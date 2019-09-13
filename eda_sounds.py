import os
import sys

import wave
import contextlib

import numpy as np

from tqdm import tqdm

import pickle

import pandas as pd

# for root,subdirs,files in os.walk("SamplePhonics") :

# Use FFT to get median or mean pitch of sample
# Use peak_amplitude/mean_amplitude to get "punchiness" / dynamics
# get length of wav file, or length of time b/w start + lowest amplitude point

# Use dummies in numpy to categorize into a few main categories
# concatenate and feed categories into a neural network to

# OR use k-means clustering to separate out samples into different types
# and recommend samples

# With clustering we can also use continuous variables - maybe dummies not necessary

from process_sound import process_sound

import matplotlib.pyplot as pp

inputs     = []
durs       = []
good_fns   = []
good_files = []

in_dir = "."
if len(sys.argv)>1 :
    in_dir = str(sys.argv[1])

npass=0
nfail=0

# keyword = "snare"
keyword = "perc"

if len(sys.argv)>2 :
    keyword = str(sys.argv[2])

for root,subdirs,files in os.walk(in_dir) :
    #print(root)
    #print(subdirs)
    #print(files)
    #raw_input()
    #print("\n\n")

    do_pause=False

    if "clusters" in root :
        continue

    # I feel like these might be massively biasing cluster selection
    if "wa_synth_drums_live_pack" in root :
        continue
    
    print("Processing: "+root)
    for f in files : #tqdm(files) :
        if ".wav" in f and keyword in f.lower() and ".asd" not in f :
            #print(root+f)
            #do_pause=True
            #nsnares+=1

            fname=root+"/"+f

            try :
                cl = contextlib.closing(wave.open(fname,'r'))
            except :
                print("Couldn't process: "+fname)
                nfail+=1
                continue
            npass+=1
            with contextlib.closing(wave.open(fname,'r')) as f0 :
                frames = f0.getnframes()
                rate = f0.getframerate()
                duration = float(frames) / float(rate)

                # Hard fix to avoid loops
                if duration>1.5 :
                    continue;

                # Derive other properties.
                # try :
                #     arrs = process_sound(fname)
                # except :
                #     continue

                arrs = process_sound(fname)
                
                # if type(arrs) != type([duration]) :
                #     continue
                # arrs.extend([duration])
                
                # Save the information
                durs.append(duration)
                inputs.append(arrs)
                #inputs.append([duration,max_freq,avg_freq,slope])
                good_fns.append(fname)
                good_files.append(f)
                
                #print(fname+" : "+str(duration))
    
print("# of "+keyword+" processed: "+str(float(npass)/float(npass+nfail)))

#pp.hist(durs,bins=list(np.arange(0,10,0.05)))
#pp.show()
#sys.exit(0)

#vls = "filename|"
vls = []
for i in range(len(inputs[0])) :
    vls.append("v"+str(i))

pvs             = pd.DataFrame(inputs)
pvs.columns     = vls
pvs['filename'] = good_fns

pickle.dump(pvs,open(keyword+".pickle","wb"))

# Let's just write out the information to file.
# with open(keyword+"_eda.csv","w") as f_out :
#     f_out.write(vls+"\n")    
#     for i in tqdm(range(len(good_fns))) :
#         f_out.write(good_fns[i]+"|")
#         for j in range(len(inputs[i])) :
#             if j<(len(inputs[i])-1) :
#                 f_out.write(str(inputs[i][j])+"|")
#             else :
#                 f_out.write(str(inputs[i][j]))
#         f_out.write("\n")
