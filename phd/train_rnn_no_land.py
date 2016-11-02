from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import time

import pdb
import os

from six.moves import xrange  
import tensorflow as tf
import resnet_rnn_no_land as resnet

import correct_224x224_dataset

UPDATE_OPS_COLLECTION = 'resnet_update_ops'


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('same', 20,'')               
flags.DEFINE_integer('different' , 4,'')               
flags.DEFINE_integer('hidden_units' , 100,'')              
flags.DEFINE_integer('d_size' , 10136,'')              ##       gia 810880 images

flags.DEFINE_integer('num_units_out_fc1',1500,'')

flags.DEFINE_integer('person_ind',0,'')
flags.DEFINE_integer('summaries',0,'')


epochs = 10
no_epochs_per_decay = 5
initial_learning_rate = 0.001
learning_rate_decay_factor = 0.97
maximum_steps = FLAGS.d_size * epochs



if FLAGS.person_ind == 1:
  inputt = '/vol/atlas/homes/dk15/person_independent_train.txt'                                 ## 902632 images
  k = '/person_independent/'
else:
  inputt = '/homes/dk15/Desktop/labels4annotators.txt'                                 ## 810880 images
  k ='/' 

fld = '/vol/atlas/homes/dk15/phd/rnn/no_land'+k+'epochs='+str(epochs)+'_same='+str(FLAGS.same)+'_different='+str(FLAGS.different)+'_hidden_units='+str(FLAGS.hidden_units)+'_lr='+str(initial_learning_rate)+'_fc1='+str(FLAGS.num_units_out_fc1)

if not os.path.exists(fld):
    os.makedirs(fld)

restore = fld+'/model.ckpt'
s_writer = fld
write_result = fld+'/error.txt'
write_prds = fld+'/predictions.txt'
write_labs = fld+'/labels.txt'

checkpoint_dir = '/homes/dk15/Desktop/tensorflow-resnet-pretrained-20160509/'

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def run_training(scope=''):
  with tf.Graph().as_default():  
    
    
    global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
    
    
    
    # Calculate the learning rate schedule.
    num_batches_per_epoch = FLAGS.same*FLAGS.different
    num_epochs_per_decay = no_epochs_per_decay
    decay_steps = num_batches_per_epoch * num_epochs_per_decay
  
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,global_step,decay_steps,learning_rate_decay_factor,staircase=True)
    
    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(lr)

    # take reshaped and cropped image and label batch size from dataset.py
    image_list, label_list = correct_224x224_dataset.read_labeled_image_list(inputt)


    images = tf.convert_to_tensor(image_list)
    labels = tf.convert_to_tensor(label_list)
  
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],num_epochs=None, shuffle=False, seed=None,capacity=10000, shared_name=None, name=None)
    image, label = correct_224x224_dataset.read_images_from_disk(input_queue,image_list)
    image = tf.to_float(image)
    image -= 128.0
    image /= 128.0

    image_same, label_same = tf.train.batch(
                          [image, label],
                          batch_size=FLAGS.same,
                          num_threads=1,
                          capacity=10000)


    image_batch, label_batch = tf.train.shuffle_batch(
                          [image_same, label_same],
                          batch_size=FLAGS.different,min_after_dequeue=1000,
                          num_threads=1,
                          capacity=10000)

    
    cnn_output = resnet.inference(image_batch,True,FLAGS.num_units_out_fc1)    
    predictions = resnet.rnn(cnn_output,FLAGS.same,FLAGS.different,FLAGS.hidden_units,FLAGS.num_units_out_fc1)    


    restorer = tf.train.Saver(tf.all_variables()[1:-2-10-2])


     
    # Add to the Graph the Ops for loss calculation.
    label_batch = tf.reshape(label_batch,[-1,8])
    loss = resnet.loss(predictions, label_batch)

    # Calculate the gradients for the batch of data
    grads = opt.compute_gradients(loss)
    
    # Another possibility is to use tf.slim.get_variables().
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    variables_to_average = (tf.trainable_variables() +
            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    tf.add_to_collection(UPDATE_OPS_COLLECTION, variable_averages.apply([loss]))
    
    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    
    #Group all updates to into a single train op.
    # NOTE: Currently we are not using batchnorm in MDM.
    train_op = tf.group(apply_gradient_op, variables_averages_op,batchnorm_updates_op)
    # Create a saver.
    saver = tf.train.Saver(tf.all_variables(),max_to_keep = epochs)
    
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config = config)
   
    sess.run(tf.initialize_all_variables())
    

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
      # Restores from checkpoint with absolute path.
          restorer.restore(sess, ckpt.model_checkpoint_path)
        else:
      # Restores from checkpoint with relative path.
          restorer.restore(sess, os.path.join(checkpoint_dir,ckpt.model_checkpoint_path))
        print('Succesfully loaded model from %s .' %(ckpt.model_checkpoint_path))
    else:
        print('No checkpoint file found')
        return

    #######SUMMARIES
  
    if FLAGS.summaries == 1:
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
      summaries.append(tf.histogram_summary('predictions', predictions))
      summaries.append(tf.histogram_summary('labels', label_batch))
      summaries.append(tf.scalar_summary('loss', loss))  
      tf.scalar_summary('loss_avg', variable_averages.average(loss))
    
      # Add a summary to track the learning rate.
      summaries.append(tf.scalar_summary('learning_rate', lr))
  
      # Add histograms for gradients.
      for grad, var in grads:
       if grad is not None:
         summaries.append(tf.histogram_summary(var.op.name + '/gradients', grad))
    
      # Add histograms for trainable variables.
      for var in tf.trainable_variables():
       summaries.append(tf.histogram_summary(var.op.name, var))
      
      # Build the summary operation from the last tower summaries.
      summary_op = tf.merge_summary(summaries)
      summary_writer = tf.train.SummaryWriter(s_writer,sess.graph)


    
      
    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
        
   
    
#    prds = []
#    labs = []
    error = [] 
    i = 0  
    with open(write_result, "w") as tr:
          
      for epoch in xrange(maximum_steps):
          
          start_time = time.time()
          duration = time.time() - start_time

          _, loss_value = sess.run([train_op, loss])

          error.append(loss_value)
          #prds.append(prs)
          #labs.append(lbs)
          if np.isnan(loss_value):
            pdb.set_trace()

          
          if epoch % 500 ==0:      
            format_str = (' step %d, loss = %.2f ( %.3f '
              'sec/batch)')
            print(format_str % ( epoch, loss_value,duration))
          
          if FLAGS.summaries == True:
            if epoch % 5000 ==0:  
             summary_str = sess.run(summary_op)
             summary_writer.add_summary(summary_str, epoch)

          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          if epoch % (FLAGS.d_size-1) == 0 and epoch != 0:
            format_str = (' step %d, loss = %.2f ( %.3f '
              'sec/batch)')
            print(format_str % ( epoch, loss_value,duration))

            saver.save(sess,restore, global_step=epoch)
          
            new_error = error[int(i*FLAGS.d_size):int(i*FLAGS.d_size+epoch)]  
            error_mean = float(sum(new_error)/len(new_error))
            tr.write(str(error_mean)+'\n') 
            i = i+1
              



def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
