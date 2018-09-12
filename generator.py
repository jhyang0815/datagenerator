import tensorflow as tf
import numpy as np
import librosa
import os
from random import shuffle
#input: 64kbps 8khz
#output: 44100hz wav


# metric = x-y (wav)

def data_generator(batch_size,x_dir,y_dir,framesize):
    #resample 된 8khz짜리랑 16khz짜리가 필요함.
    fnames=os.listdir(x_dir)
    #print(fnames)
    while True:
        shuffle(fnames)
        for fname in fnames:
            data_x=np.load(os.path.join(x_dir,fname))
            data_y=np.load(os.path.join(y_dir,fname))
            frame_batch_x = []
            frame_batch_y = []
            for i in range(len(data_x)//framesize):
                frame_x=data_x[i*framesize:(i+1)*framesize]
                frame_y = data_y[i * framesize:(i + 1) * framesize]
                frame_batch_x.append(frame_x)
                frame_batch_y.append(frame_y)
                if len(frame_batch_x)==batch_size:
                    frame_batch_x=np.array(frame_batch_x)
                    frame_batch_y=np.array(frame_batch_y)
                    frame_batch_x=frame_batch_x[:,:,np.newaxis]
                    frame_batch_y = frame_batch_y[:, :, np.newaxis]
                    yield frame_batch_x,frame_batch_y
                    frame_batch_x = []
                    frame_batch_y = []

def data_generator_shift(batch_size,x_dir,y_dir,framesize,frameshift):
    
    fnames=os.listdir(x_dir)
    #print(fnames)
    while True:
        shuffle(fnames)
        for fname in fnames:
            data_x=np.load(os.path.join(x_dir,fname))
            data_y=np.load(os.path.join(y_dir,fname))
            frame_batch_x = []
            frame_batch_y = []
            for i in range(len(data_x)//(framesize-frameshift)-1):
                frame_x=data_x[i*frameshift:i*frameshift+framesize]
                frame_y = data_y[i*frameshift:i*frameshift+framesize]
                frame_batch_x.append(frame_x)
                frame_batch_y.append(frame_y)
                if len(frame_batch_x)==batch_size:
                    frame_batch_x=np.array(frame_batch_x)
                    frame_batch_y=np.array(frame_batch_y)
                    frame_batch_x=frame_batch_x[:,:,np.newaxis]
                    frame_batch_y = frame_batch_y[:, :, np.newaxis]
                    yield frame_batch_x,frame_batch_y
                    frame_batch_x = []
                    frame_batch_y = []

