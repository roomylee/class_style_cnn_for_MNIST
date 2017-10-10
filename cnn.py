import numpy as np
import tensorflow as tf

class ConvNet:
    # truncated normal distribution에 기반해서 랜덤한 값으로 초기화
    def weight_variable(shape):
        # tf.truncated_normal:
        # Outputs random values from a truncated normal distribution.
        # values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 0.1로 초기화
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # convolution & max pooling
    # vanila version of CNN
    # x (아래 함수들에서) : A 4-D `Tensor` with shape `[batch, height, width, channels]`
    def conv2d(x, W):
        # stride = 1, zero padding은 input과 output의 size가 같도록.
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        # pooling
        # [[0,3],
        #  [4,2]] => 4

        # [[0,1],
        #  [1,1]] => 1
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def __init__(self):
        # [5,5,1,32]: 5x5 convolution patch, 1 input channel, 32 output channel.
        # MNIST의 pixel은 0/1로 표현되는 1개의 벡터이므로 1 input channel임.
        # CIFAR-10 같이 color인 경우에는 RGB 3개의 벡터로 표현되므로 3 input channel일 것이다.
        # Shape을 아래와 같이 넣으면 넣은 그대로 5x5x1x32의 텐서를 생성함.
        W_conv1 = ConvNet.weight_variable([5, 5, 1, 32])
        b_conv1 = ConvNet.bias_variable([32])
        # 최종적으로 32개의 output channel에 대해 각각 5x5의 convolution patch(filter) weight와 1개의 bias를 갖게 됨.

        # x는 [None, 784] (위 placeholder에서 선언). 이건 [batch, 28*28] 이다.
        # x_image는 [batch, 28, 28, 1] 이 됨. -1은 batch size를 유지하는 것이고 1은 color channel.
        # (40000,784) => (40000,28,28,1)
        image = tf.reshape(x, [-1, image_width, image_height, 1])
        # print (image.get_shape()) # =>(40000,28,28,1)


        h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
        # print (h_conv1.get_shape()) # => (40000, 28, 28, 32)
        h_pool1 = ConvNet.max_pool_2x2(h_conv1)
        # print (h_pool1.get_shape()) # => (40000, 14, 14, 32)


        # Prepare for visualization
        # display 32 fetures in 4 by 8 grid
        layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4, 8))

        # reorder so the channels are in the first dimension, x and y follow.
        layer1 = tf.transpose(layer1, (0, 3, 1, 4, 2))

        layer1 = tf.reshape(layer1, (-1, image_height * 4, image_width * 8))