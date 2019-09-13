import os
import sys

import wave
import contextlib

import numpy as np

from tqdm import tqdm

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

import pickle

# Let's do some kmeans clustering on duration.
from sklearn.cluster import KMeans

import pandas as pd

from sklearn.metrics import silhouette_samples, silhouette_score

keyword = "perc"

if len(sys.argv)>1 :
    keyword = str(sys.argv[1])

#data     = pd.read_csv(keyword+"_eda.csv",delimiter="|").fillna(0.)
#data     = data[~data.filename.str.contains("-2.wav")]

data = pickle.load(open(keyword+".pickle","rb"))
data = data.replace([np.inf, -np.inf], np.nan).dropna(how="all")

data       = data.fillna(0)
data_cts   = data.iloc[:,0:-1]
good_fns   = []
good_files = []

X = []
#for d in durs :
#    X.append([d])
for ii in range(len(data)) :
    
    fp = str(data.iloc[ii]['filename'])
    fn = fp.split("/")[-1]

    good_fns.append(fp)
    good_files.append(fn)
    X.append(data.iloc[ii,1:].tolist())
    
#nclus  = 10
#kmeans = KMeans(n_clusters=nclus, random_state=0).fit(np.array(X))

# Import required packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
    
mms = MinMaxScaler()
mms.fit(data_cts)
#data_transformed = mms.transform(data_cts)

# Let's take off the scaling for a sec.
data_transformed = data.iloc[:,0:-1]
    
# Try the elbow method to select the optimal # of clusters.
# Sum_of_squared_distances = []
# sil_avgs = []
# K = range(2,20)
# for k in K:
#     km = KMeans(n_clusters=k)
#     km = km.fit(data_transformed)
#     Sum_of_squared_distances.append(km.inertia_)
#     cluster_labels = km.fit_predict(data_transformed)
#     silhouette_avg = silhouette_score(data_transformed, cluster_labels)
#     sil_avgs.append(silhouette_avg)
#     print("For n_clusters = "+str(k)+", the average silhouette_score is :"+str(silhouette_avg))

# nmax_sa = np.argmax(sil_avgs)
# print("Optimal # of clusters is "+str(list(K)[nmax_sa])+" based on silhouette avg: "+str(sil_avgs[nmax_sa]))
    
# plt.plot(K, Sum_of_squared_distances, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Sum_of_squared_distances')
# plt.title('Elbow Method For Optimal k')
# plt.show()

############################################################
# Let's arrange files according to clusters!
############################################################

print("How many clusters would you like to make?")
nclus=int(raw_input())
print("Making "+str(nclus)+" clusters from the data")
# kmeans = KMeans(n_clusters=nclus, random_state=0).fit(data_transformed)

kmeans = KMeans(n_clusters=nclus).fit(data_transformed)

#print(kmeans.cluster_centers_)
#centers = np.array(kmeans.cluster_centers_)
#print(centers.shape)
#plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')
#plt.show()

try :
    os.mkdir("clusters")
except :
    print("clusters folder exists")

os.system("rm -rf clusters/"+keyword)
os.mkdir("clusters/"+keyword)
for i in range(nclus) :
    os.mkdir("clusters/"+keyword+"/"+str(i))
    
# for i in tqdm(range(len(good_fns))) :
for i in range(len(good_fns)) :
    ic = kmeans.predict([data_transformed.iloc[i]])
    #print(good_fns[i]+" : "+str(ic))
    #print("Operating on "+good_fns[i]+"with cluster "+str(ic))
    #print('ln -s "${PWD}/'+good_fns[i]+'" "${PWD}/clusters/snare/'+str(ic[0])+'/'+good_files[i]+'"')
    os.system('ln -s "${PWD}/'+good_fns[i]+'" "${PWD}/clusters/'+keyword+'/'+str(ic[0])+'/'+good_files[i]+'"')

# Save the kmeans and projected vectors to file.
pickle.dump(kmeans,open(keyword+"_kmeans.pickle","wb"))
