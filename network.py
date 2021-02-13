import tensorflow as tf

def network(input, weights, biases, strides = 1):
    conv1 = tf.nn.conv2d(input, weights['w_c_1'], strides=[1, strides, strides, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, biases['b_c_1'])
    relu1 = tf.nn.relu(conv1)

    conv2 = tf.nn.conv2d(relu1, weights['w_c_2'], strides=[1, strides, strides, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, biases['b_c_2'])
    relu2 = tf.nn.relu(conv2)

    conv3 = tf.nn.conv2d(relu2, weights['w_c_3'], strides=[1, strides, strides, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, biases['b_c_3'])
    relu3 = tf.nn.relu(conv3)

    conv4 = tf.nn.conv2d(relu3, weights['w_c_4'], strides=[1, strides, strides, 1], padding='SAME')
    conv4 = tf.nn.bias_add(conv4, biases['b_c_4'])
    relu4 = tf.nn.relu(conv4)

    sum_l = tf.reduce_sum(relu4, axis=2)
    max_l = tf.reduce_max(relu4, axis=2)

    concat = tf.concat([sum_l, max_l], axis=2)

    h1 = tf.tensordot(concat, weights['w_fc_1'], axes = [[2], [0]]) #tf.tensordot(a, b, axes)
    h1 = tf.add(h1, biases['b_fc_1'])
    h1 = tf.nn.relu(h1)

    h2 = tf.tensordot(h1, weights['w_fc_2'], axes=[[2], [0]])
    h2 = tf.add(h2, biases['b_fc_2'])
    h2 = tf.nn.relu(h2)

    out = tf.tensordot(h2, weights['w_fc_3'] ,axes = [[2], [0]])
    out = tf.add(out, biases['b_fc_3'])
    pred = tf.nn.sigmoid(out)

    return pred







