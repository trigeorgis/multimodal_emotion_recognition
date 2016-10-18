from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import time

import pdb

from six.moves import xrange  
import tensorflow as tf



import os

from menpo.transform import Translation, GeneralizedProcrustesAnalysis
import menpo.io  as mio
from menpo.visualize import print_progress


# structure of landmark files:  /videoname/*.pts where names of pts files are numbers starting from 0

# input content is a sorted list of all videonames 
def run_training(content):
    
   

    # for every file that has landmarks import them
    shapes = []
    for elem in content:
     shapes.append(mio.import_landmark_files((elem+'/').rstrip()))

    centered_shapes = []
    video_no_of_landmarks = []
    for i in range(len(shapes)):   # for all videos/folders
      print(i)
      for j in range(len(shapes[i])):  # for every landmark
         try:        
            s = shapes[i][j]
            if s.n_landmarks==64 and not np.any(np.isnan(s.lms.points)):
              s = Translation(-s.lms.centre()).apply(s.lms)
            centered_shapes.append(s)
         except:
            print('except')
            pass
      # because centered_shapes have all landmarks we need to store how many landmarks each video has because we will need it later      
      video_no_of_landmarks.append(j)
    
    # align centralized shape using Procrustes Analysis
   
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    aligned_shapes = [s.aligned_source() for s in gpa.transforms]
    counter = 0
    k = 0
    
    for elem in content:  # for every vid
      elem = elem.rstrip()
      for i in range(video_no_of_landmarks[counter]):  # for every landmark in a vid
         land_vector = aligned_shapes[i+k].as_vector() # flatten each aligned shape so we have a vector of coordinates of a landmark
         _,vid = elem[:].split('/vol/atlas/homes/dk15/full_landmarks/')
         
         # write landmarks to csv file with lines having coordinates (x,y) of a landmark point 
         with open('/vol/atlas/homes/dk15/landmarks/gen_proc/'+vid+str(i)+'.csv','w') as f:
            for x,y in zip(*[iter(land_vector)]*2): # for all points (pair x,y) of a landmark
              f.write(str(x)+','+str(y)+'\n')
      k += i # total number of landmarks till this point
      counter += 1  # go to next vid (vid no=101-500 counter no=0-499)
    pdb.set_trace()
      

def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
