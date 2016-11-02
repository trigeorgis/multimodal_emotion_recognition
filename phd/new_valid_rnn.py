from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import time

import pdb
import os

from six.moves import xrange  
import tensorflow as tf


import resnet_rnn as resnet
import correct_224x224_dataset

UPDATE_OPS_COLLECTION = 'resnet_update_ops'

epochs = 1
same = 25
different = 3 
d_size = 2700 
hidden_units = 150


maximum_steps = d_size * epochs

inputt = '/homes/dk15/Desktop/labels4annotators_valid.txt'
input_land = '/homes/dk15/Desktop/lands4annotators_valid.txt'

fld = '/vol/atlas/homes/dk15/phd/rnn/epochs=10_lr=0.001_same=25_different=3/valid_10811_shuffle=false'
 
restore = fld+'/model.ckpt'
s_writer = fld
write_result = fld+'/error.txt'
write_prds = fld+'/predictions.txt'
write_labs = fld+'/labels.txt'

checkpoint_dir = '/vol/atlas/homes/dk15/phd/rnn/epochs=10_lr=0.001_same=25_different=3'


# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999




def run_training(scope=''):
  with tf.Graph().as_default():  
    
    

    # take reshaped and cropped image and label batch size from dataset.py
    image_list, label_list = correct_224x224_dataset.read_labeled_image_list(inputt)
    land_list = correct_224x224_dataset.read_labeled_land_list(input_land)

    images = tf.convert_to_tensor(image_list)
    labels = tf.convert_to_tensor(label_list)
    lands = tf.convert_to_tensor(land_list)
  
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels,lands],num_epochs=None, shuffle=False, seed=None,capacity=10000, shared_name=None, name=None)
    image, label,land = correct_224x224_dataset.read_images_lands_from_disk(input_queue)
    image = tf.to_float(image)
    image -= 128.0
    image /= 128.0
    
    land /=  108.28557307125897   # min = -108.28557307125897, max = 106.92839188558608

    image_same, label_same,land_same = tf.train.batch(
                          [image, label,land],
                          batch_size=same,
                          num_threads=1,
                          capacity=10000)


    image_batch, label_batch,land_batch = tf.train.batch(
                          [image_same, label_same,land_same],
                          batch_size=different,
                          num_threads=1,
                          capacity=10000)

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
   
    cnn_output = resnet.inference(image_batch,land_batch,False)    
    predictions = resnet.rnn(cnn_output,same,different,hidden_units)    


    summaries.append(tf.histogram_summary('predictions', predictions))
     
    # Add to the Graph the Ops for loss calculation.
    label_batch = tf.reshape(label_batch,[-1,8])

    summaries.append(tf.histogram_summary('labels', label_batch))

    loss0 = resnet.concordance_loss(predictions[:,0], label_batch[:,0])
    loss1 = resnet.concordance_loss(predictions[:,1], label_batch[:,1])
    loss2 = resnet.concordance_loss(predictions[:,2], label_batch[:,2])
    loss3 = resnet.concordance_loss(predictions[:,3], label_batch[:,3])
    loss4 = resnet.concordance_loss(predictions[:,4], label_batch[:,4])
    loss5 = resnet.concordance_loss(predictions[:,5], label_batch[:,5])
    loss6 = resnet.concordance_loss(predictions[:,6], label_batch[:,6])
    loss7 = resnet.concordance_loss(predictions[:,7], label_batch[:,7])

    mse =  resnet.loss(predictions, label_batch)

    total_loss = (loss0+loss7+loss1+loss2+loss6+loss5+loss4+loss3)/8 

    summaries.append(tf.scalar_summary('loss', total_loss))

  
    
    # Another possibility is to use tf.slim.get_variables().
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    # Create a saver.
    restorer = tf.train.Saver(variables_to_restore,max_to_keep= epochs)

    
            
    # Build the summary operation from the last tower summaries.
    summary_op = tf.merge_summary(summaries)

    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config = config)
   

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
      # Restores from checkpoint with absolute path.
          restorer.restore(sess, ckpt.model_checkpoint_path)
        else:
      # Restores from checkpoint with relative path.
          restorer.restore(sess, os.path.join(checkpoint_dir,ckpt.model_checkpoint_path))
          
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s .' %(ckpt.model_checkpoint_path))
    else:
        print('No checkpoint file found')
        return
    
      
    
      
  # Start the queue runners.
    coord = tf.train.Coordinator()
    try:

      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))
                              
            
       
      summary_writer = tf.train.SummaryWriter(s_writer,sess.graph)
    
      in_queue = []
      prds = []
      labs = []
      error = [] 
      i = 0  
      mse_loss = []
      for epoch in xrange(maximum_steps):
          
       
          loss_value,prs,lbs,mse_l = sess.run([total_loss,predictions,label_batch,mse])
          error.append(loss_value)
          prds.append(prs)
          labs.append(lbs)
          mse_loss.append(mse_l)
          if np.isnan(loss_value):
            pdb.set_trace()

          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, epoch)


          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        
      error_mean= float(sum(error)/len(error))
      mse_mean= float(sum(mse_loss)/len(mse_loss))


      ppr = np.reshape(np.vstack(prds).ravel(), (-1, 1))
      plabs = np.reshape(np.vstack(labs).ravel(), (-1, 1))
      pos_pr = sum(1 for no in ppr if no >0)
      neg_pr = sum(1 for no in ppr if no<0)
      pos_lab = sum(1 for no in plabs if no >0)
      neg_lab = sum(1 for no in plabs if no<0)
        
        
              

      with open(write_result, "w") as tr:
                tr.write('error mean= '+str(error_mean)+'\n') 
                tr.write('mse mean= '+str(mse_mean)+'\n') 

        
      with open(write_prds, "w") as aa:
                for line in prds:
                    aa.write(str(line)+'\n')
                aa.write('\n')
                aa.write('\n')
                aa.write('positives= '+str(pos_pr)+'\n')
                aa.write('negatives= '+str(neg_pr)+'\n')
      with open(write_labs, "w") as bb:
                for line in labs:
                    bb.write(str(line)+'\n')        
                bb.write('\n')
                bb.write('\n')
                bb.write('positives= '+str(pos_lab)+'\n')
                bb.write('negatives= '+str(neg_lab)+'\n')      
    
    except Exception as e:  # pylint: disable=broad-except
            print('o coordinator FTAIEIIIIIIIIIIIIIIIIII')
            coord.request_stop(e)

    finally:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=3) 




def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
