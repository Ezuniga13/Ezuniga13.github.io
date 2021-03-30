---
title: "Playing Wav files for sanity check in Python"
date: 2021-03-29
tags: [Python, data science,]
header:
  image: "https://images.pexels.com/photos/691467/art-smoke-light-lights-691467.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=500"
excerpt: "Deep Learning, Librosa, Audio Classification"
mathjax: "true"
---







#### This post is the first in a series to present a classification project using deep learning. ####

Playing wav files for a sanity check.

1. import libraries 
2. create paths to wav files
3. play audio of a wav file
4. display the wavplot using librosa libray 

Here is the libraries we will need to play the sounds and to visualize them.


```python
import IPython.display as ipd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
%matplotlib inline
```

Here is how we create paths to the downloaded files.


```python
path = os.getcwd()
file_name = '/cats_dogs/train/cat/'
new_path = os.path.join(path, file_name)
os.chdir(new_path)
```

Please keep in mind that the file name will vary depending on where the file is located in your local machine.
Next will create a variable in order for the `ipd.Audio` to play the file.


```python
cats = os.listdir('/Users/estebanzuniga/Project-5/cats_dogs/train/cat/')

```

Using IPython.display. Audio we can play the audio file. (33 is just the index of this particular file)


```python
ipd.Audio(cats[33])
```





Next let's use the librosa package to visualize what the sound wave looks like.
Librosa is a Python package for music and audio processing. 
X refers to the length of the array and SR stands for simple rate. 
Here we a loading the audio file into an audio array.


```python
x, sr = librosa.load(cats[20], sr = None)
len(x), sr
```




    (30583, 16000)



Now for the fun stuff. Here we will plot what the audio file looks like. For that will use and mix of matplolib and librosa


```python
plt.figure()
librosa.display.waveplot(x, sr = sr)
plt.xlabel("Time (seconds)-->")
plt.ylabel("amplitude")
```




    Text(0, 0.5, 'amplitude')




![png](Portfolio_Post_files/Portfolio_Post_11_1.png)


#### In the next post we will convert these visualizations into spectrograms, which wil be necessary to in order to classify the audio using images instead of sound. ####
