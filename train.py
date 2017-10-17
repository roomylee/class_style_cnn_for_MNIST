import tensorflow as tf
import datetime
from cnn import ConvNet
import data_helpers


def train(batch_size=50, epochs=2500, dropout=0.5):
    train_images, train_labels = data_helpers.load_data_and_labels('data/train.csv')

    # input & output of NN
    # images
    input_x = tf.placeholder(tf.float32, shape=[None, 28*28], name='input_x')
    # labels
    input_y = tf.placeholder(tf.float32, shape=[None, 10], name='input_y')
    # dropout_keep_prob은 dropout을 적용할지 말지에 대한 확률임. 이를 이용해서 training 동안만 드롭아웃을 적용하고 testing 때는 적용하지 않는다.
    # training & evaluation 코드를 보니 keen_prob = 1.0일때 dropout off 인 듯.
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    model = ConvNet(input_x, input_y, dropout_keep_prob)

    # start TensorFlow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    display_step = 1

    with sess.as_default():
        batches = data_helpers.batch_iter(list(zip(train_images, train_labels)), batch_size, epochs)
        for step, batch in enumerate(batches):
            batch_x, batch_y = zip(*batch)
            loss, accuracy = sess.run([model.cross_entropy, model.accuracy],
                                      feed_dict={input_x: batch_x, input_y: batch_y, dropout_keep_prob: 1.0})
            if step % display_step == 0 or (step + 1) == epochs:
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if step % (display_step * 10) == 0 and step:
                    display_step *= 10
            # train on batch
            sess.run(model.train_step, feed_dict={input_x: batch_x, input_y: batch_y, dropout_keep_prob: dropout})


def main():
    train()

if __name__ == "__main__":
    tf.app.run()
