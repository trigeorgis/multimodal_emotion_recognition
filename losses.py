import tensorflow as tf

def concordance_cc(predictions, labels):
    
    pred_mean, pred_var = tf.nn.moments(predictions, (0,))
    gt_mean, gt_var = tf.nn.moments(labels, (0,))

    mean_cent_prod = tf.reduce_mean((predictions - pred_mean) * (labels - gt_mean))

    return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))