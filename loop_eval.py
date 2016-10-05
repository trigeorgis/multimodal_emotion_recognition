import tensorflow as tf
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
from menpo.visualize import print_progress
slim = tf.contrib.slim
from time import sleep
import valid2


while True:
    valid2.valid()
    sleep(600)
