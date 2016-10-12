#matplotlib inline
import tensorflow as tf
import dataset_rec_state
import model_rec_state 
import numpy as np
import losses
from menpo.visualize import print_progress
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(checkpoint_dir,'/vol/atlas/homes/dk15/full_recola/new/mse(0.0001,4)',
                          'where is the model checkpoint')
tf.app.flags.DEFINE_string(portion,'valid',
                          'what set are we evaluating')
tf.app.flags.DEFINE_integer(total_portion_vids,15,
                          'how many vids we are evaluating')

nogpu_config = tf.ConfigProto(
    # Do not use a GPU device
    device_count = {'GPU': 0}
)

sess = tf.Session(config=nogpu_config)
audio, ground_truth = dataset_rec_state.get_split(FLAGS.portion)

with tf.variable_scope('net'):
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                        is_training=False):
        prediction,states = model_rec_state.audio_model(audio)
# initialse the variable for the states
#init_op = tf.initialize_variables(tf.all_variables(),'net/Variable')
#sess.run(init_op)

variables_to_restore = slim.get_variables_to_restore()
# since we didnt train with the state variable, we cannot restore it
#variables_to_restore = variables_to_restore[0:4]+variables_to_restore[5:]

saver = tf.train.Saver(variables_to_restore)
model_path = slim.evaluation.tf_saver.get_checkpoint_state(FLAGS.checkpoint_dir).model_checkpoint_path
saver.restore(sess, model_path)

print(model_path)

_ = tf.train.start_queue_runners(sess=sess)
predictions = []
ground_truths = []

for i in print_progress(range(FLAGS.total_portion_vids)):
    pred, ground__truth = sess.run([prediction, ground_truth])
    predictions.append(pred)
    ground_truths.append(ground__truth)



predictions_flattened = np.reshape(predictions,[-1,2])
labels_flattened = np.reshape(gts,[-1,2])
mse_arousal = losses.mse(predictions_flattened[:,0],labels_flattened[:,0])
mse_valence = losses.mse(predictions_flattened[:,1],labels_flattened[:,1]) 
mse = losses.mse(predictions_flattened,labels_flattened) 
print(mse_valence)
print(mse_arousal)
print(mse)
print(losses.concordance_cc2(predictions_flattened[:,0],labels_flattened[:,0]))
print(losses.concordance_cc2(predictions_flattened[:,1],labels_flattened[:,1]))
print((losses.concordance_cc2(predictions_flattened[:,0],labels_flattened[:,0])+losses.concordance_cc2(predictions_flattened[:,1],labels_flattened[:,1]))/2)

print()
print()

#plt.plot(predictions[0][..., 0].ravel())
#plt.plot(gts[0][..., 0].ravel())
#concordance_cc(np.array(predictions)[..., 0].ravel(), np.array(gts)[..., 0].ravel())
