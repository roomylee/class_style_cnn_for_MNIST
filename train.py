import numpy as np
import pandas as pd
import tensorflow as tf
from cnn import ConvNet

# image number to output
IMAGE_TO_DISPLAY = 10

# read training data from CSV file
data = pd.read_csv('data/train.csv')

print('data({0[0]},{0[1]})'.format(data.shape))
print(data.head())

images = data.iloc[:, 1:].values
images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
print('images({0[0]},{0[1]})'.format(images.shape))

image_size = images.shape[1]
print('image_size => {0}'.format(image_size))

# in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint16)
print('image_width => {0}\nimage_height => {1}'.format(image_width, image_height))

labels_flat = data['label'].values.ravel()
print('labels_flat({0})'.format(len(labels_flat)))
print('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY, labels_flat[IMAGE_TO_DISPLAY]))

labels_count = np.unique(labels_flat).shape[0]
print('labels_count => {0}'.format(labels_count))



# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(labels.shape))
print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))

train_images = images
train_labels = labels
print('train_images({0[0]},{0[1]})'.format(train_images.shape))



# settings
LEARNING_RATE = 1e-4
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 2500

DROPOUT = 0.5
BATCH_SIZE = 50


epochs_completed = 0
index_in_epoch = 0
num_examples = images.shape[0]


# serve data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


model = ConvNet(image_width, image_height, labels_count)

# start TensorFlow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


display_step = 1

with sess.as_default():
    for i in range(TRAINING_ITERATIONS):
        # get new batch
        batch_xs, batch_ys = next_batch(BATCH_SIZE)

        # check progress on every 1st,2nd,...,10th,20th,...,100th... step
        if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:

            train_accuracy = model.accuracy.eval(feed_dict={model.x: batch_xs,
                                                      model.y_: batch_ys,
                                                      model.keep_prob: 1.0})
            print('training_accuracy => %.4f for step %d' % (train_accuracy, i))

            # increase display_step
            if i % (display_step * 10) == 0 and i:
                display_step *= 10
        # train on batch
        sess.run(model.train_step, feed_dict={model.x: batch_xs, model.y_: batch_ys, model.keep_prob: DROPOUT})
