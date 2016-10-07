#matplotlib inline
import tensorflow as tf
import data_provider as dd
import model_rec_state as m
import numpy as np
from menpo.visualize import print_progress
slim = tf.contrib.slim


nogpu_config = tf.ConfigProto(
    # Do not use a GPU device
    device_count = {'GPU': 0}
)

sess = tf.Session(config=nogpu_config)
audio, ground_truth = dd.get_split('valid')

with tf.variable_scope('net'):
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                        is_training=False):
        prediction,states = m.audio_model(audio)
# initialse the variable for the states
init_op = tf.initialize_variables(tf.all_variables(),'net/Variable')
sess.run(init_op)

variables_to_restore = slim.get_variables_to_restore()
# since we didnt train with the state variable, we cannot restore it
variables_to_restore = variables_to_restore[0:4]+variables_to_restore[5:]

saver = tf.train.Saver(variables_to_restore)
model_path = slim.evaluation.tf_saver.get_checkpoint_state('/vol/atlas/homes/dk15/full_recola/new/mse(0.0001,4)').model_checkpoint_path
saver.restore(sess, model_path)

print(model_path)

_ = tf.train.start_queue_runners(sess=sess)
predictions = []
gts = []

for i in print_progress(range(50)):
    p, gt,sstates = sess.run([prediction, ground_truth,states])
    predictions.append(p)
    gts.append(gt)
    s.append(s_ids)

def concordance_cc(r1, r2):
    mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()

    return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)

pr = np.reshape(predictions,[-1,2])
lb = np.reshape(gts,[-1,2])
mse_a = ((pr[:,0]-lb[:,0])**2).mean() 
mse_v = ((pr[:,1]-lb[:,1])**2).mean() 
mse = ((pr-lb)**2).mean() 
print(mse_v)
print(mse_a)
print(mse)
print(concordance_cc(pr[:,0],lb[:,0]))
print(concordance_cc(pr[:,1],lb[:,1]))
print((concordance_cc(pr[:,0],lb[:,0])+concordance_cc(pr[:,1],lb[:,1]))/2)

print()
print()

#plt.plot(predictions[0][..., 0].ravel())
#plt.plot(gts[0][..., 0].ravel())
#concordance_cc(np.array(predictions)[..., 0].ravel(), np.array(gts)[..., 0].ravel())
