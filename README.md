# ClusteringMusicSamples

Like many other bedroom musicians, I have too many drum samples. These
samples are variable in tone, timbre, and duration; some are tight snares,
some are long, booming kick drums, some are metallic, twangy
percussion.

Given the relatively large sample library I've amassed, I
figured I'd try applying k-means clustering to sort my large, messy
collection of music samples into similar-sounding groups, to the point
that I could draw up a given sound and find similar or complimentary
sounds in my library.

**Note:** This codebase was basically hacked together in a weekend as
  an experiment, so it's not very nicely- or intuitively-structured. I
  may return to it in the future to rewrite the codebase in a nicer,
  more object-oriented fashion.

## process_sound.py

This script reads in a WAV file and produces binned distributions of
amplitude and frequency, . Both amplitude and frequency profiles serve as inputs to the clustering
algorithm, e.g. a short, snappy kick drum will sound
quite different from a crash cymbal in both pitch and amplitude.

## eda_sounds.py

This script walks through a directory structure, looking for files
with a specific keyword (e.g. "snare") and uses the `process_sound`
function to generate amplitude and frequency profiles of the
sound. After converting all detected files, the profiles are written
to a pickle file for later use in clustering.

## kmeans.py

This script reads in the created pickle file and organizes the sounds
into a user-defined number of clusters, where the input feature vector
are essentially the concatenation of the amplitude and pitch profiles.

I haven't spent much effort
trying to optimize the algorithm, as I've found that the usual
approach of optimizing with respect to the average silhouette score
(i.e. finding the "elbow" of the SSE curve) tends to create a fairly
low number of clusters. Regardless, the script will output softlinks
to each of the sounds, organized according to cluster.

In the future I might build out this portion of the code to allow the
user to find e.g. the n-nearest sounds to an arbitrary input sound. 
