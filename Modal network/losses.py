import tensorflow as tf

slim = tf.contrib.slim

def concordance_cc(predictions, labels):
    
    pred_mean, pred_var = tf.nn.moments(predictions, (0,))
    gt_mean, gt_var = tf.nn.moments(labels, (0,))

    mean_cent_prod = tf.reduce_mean((predictions - pred_mean) * (labels - gt_mean))

    return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))

def concordance_cc2(r1, r2):
    mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()

    return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)

def mse(pr,lab):
    return ((pr-lb)**2).mean() 

def get_losses(prediction,ground_truth):
	for i, name in enumerate(['arousal', 'valence']):
			pred_single = tf.reshape(prediction[:, :, i], (-1,))
			gt_single = tf.reshape(ground_truth[:, :, i], (-1,))

			loss = concordance_cc(pred_single, gt_single)
			tf.scalar_summary('losses/{} loss'.format(name), loss)
			mse = tf.reduce_mean(tf.square(pred_single - gt_single))
			tf.scalar_summary('losses/rmse {} loss'.format(name), mse)

			slim.losses.add_loss(loss / 2.)
	
	return slim.losses.get_total_loss()
