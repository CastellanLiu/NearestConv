import tensorflow as tf

def nearest_conv(x, out_channels, kernel):
    # kernel: [3, 3, out_channels, in_channels]
    w = tf.pad(kernel, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [x.shape[0], out_channels, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2],
                                  padding='SAME', data_format='NCHW')
