import caffe
import numpy as np
import yaml
from numpy import savetxt
import os
from npy_append_array import NpyAppendArray


class MyLayer(caffe.Layer):

    def setup(self, bottom, top):
       pass

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top[0].reshape(*bottom[0].shape)
        top[0].data[...] = bottom[0].data
        #print(top[0].data[...].shape)
        #np.save('data.npy',top[0].data[...])
        #print(top[0].data[...])
        filename ='Input/Inference/PP/Data/56_M1.npy'
        with NpyAppendArray(filename) as npaa:
            npaa.append(top[0].data[...])
   
          

    def backward(self, top, propagate_down, bottom):
        pass





