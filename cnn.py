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

    def __init__(self, input_width, input_height, labels_count):
        # input & output of NN
        # images
        self.x = tf.placeholder('float', shape=[None, input_width*input_height])
        # labels
        self.y_ = tf.placeholder('float', shape=[None, labels_count])


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
        image = tf.reshape(self.x, [-1, input_width, input_height, 1])
        # print (image.get_shape())
        # =>(40000,28,28,1)

        h_conv1 = tf.nn.relu(ConvNet.conv2d(image, W_conv1) + b_conv1)
        # print (h_conv1.get_shape())
        # => (40000, 28, 28, 32)
        h_pool1 = ConvNet.max_pool_2x2(h_conv1)
        # print (h_pool1.get_shape())
        # => (40000, 14, 14, 32)


        # second convolutional layer
        # channels (features) : 32 => 64
        # 5x5x32x64 짜리 weights.
        W_conv2 = ConvNet.weight_variable([5, 5, 32, 64])
        b_conv2 = ConvNet.bias_variable([64])

        h_conv2 = tf.nn.relu(ConvNet.conv2d(h_pool1, W_conv2) + b_conv2)
        # print (h_conv2.get_shape()) # => (40000, 14,14, 64)
        h_pool2 = ConvNet.max_pool_2x2(h_conv2)
        # print (h_pool2.get_shape()) # => (40000, 7, 7, 64)



        # densely connected layer (fully connected layer)
        # 7*7*64는 h_pool2의 output (7*7의 reduced image * 64개의 채널). 1024는 fc layer의 뉴런 수.
        W_fc1 = ConvNet.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = ConvNet.bias_variable([1024])

        # (40000, 7, 7, 64) => (40000, 3136)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) # -1은 batch size를 유지하는 것.
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # print (h_fc1.get_shape()) # => (40000, 1024)


        # dropout
        # keen_prob은 dropout을 적용할지 말지에 대한 확률임. 이를 이용해서 training 동안만 드롭아웃을 적용하고 testing 때는 적용하지 않는다.
        #  training & evaluation 코드를 보니 keen_prob = 1.0일때 dropout off 인 듯.
        self.keep_prob = tf.placeholder('float')
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)


        # readout layer for deep net
        W_fc2 = ConvNet.weight_variable([1024, labels_count])
        b_fc2 = ConvNet.bias_variable([labels_count])
        self.y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        # print (y.get_shape()) # => (40000, 10)


        # cost function
        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))

        # optimisation function
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

        # evaluation
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'))

        # prediction function
        # [0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
        self.predict = tf.argmax(self.y, 1)