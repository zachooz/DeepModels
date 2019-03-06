import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model:

    def __init__(self, image, label):
        self.image = image
        self.learning_rate = 1e-3
        self.label = label
        self.output_dim = int(self.label.shape[1])
        self.batch_size = int(self.image.shape[0])
        self.c_image = tf.reshape(image, [self.batch_size, 28, 28, 1])

        # Output size with VALID padding math.ciel(input_size - filter_size + 1 / stride)
        # Output size with SAME padding math.ciel(input_size / stride)

        self.prediction = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()
        self.accuracy = self.accuracy_function()

    def forward_pass(self):
        """
        :return: the predicted label as a tensor
        """
        flts = tf.Variable(tf.truncated_normal([4, 4, 1, 16], stddev=0.1))
        cOut = tf.nn.conv2d(self.c_image, flts, [1, 1, 1, 1], "SAME")
        cOut = tf.nn.max_pool(cOut, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        b = tf.Variable(tf.truncated_normal([16], stddev=0.1))
        # SAME padding math.ciel(input_size / stride) = 28/2 = 14
        cOut = tf.nn.relu(tf.add(cOut, b))

        # [height, width, channels, number of filters] output of previous 16 filters = 16 channels
        flts2 = tf.Variable(tf.truncated_normal([2, 2, 16, 32], stddev=0.1))
        cOut2 = tf.nn.conv2d(cOut, flts2, [1, 1, 1, 1], "SAME")
        cOut2 = tf.nn.max_pool(cOut2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        # SAME padding math.ciel(input_size / stride) = 14/2 = 7
        # convert to non conv network, size width * height * num_channels =  7 * 7 * 32
        cOut2 = tf.reshape(cOut2, [self.batch_size, 1568])

        W3 = tf.Variable(tf.truncated_normal([1568, self.output_dim], stddev=0.1))
        b3 = tf.Variable(tf.truncated_normal([self.output_dim], stddev=0.1))
        logits = tf.add(tf.matmul(cOut2, W3), b3)

        return logits

    def loss_function(self):
        """
        :return: the loss of the model as a tensor
        """
        return tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=self.prediction)

    def optimizer(self):
        """
        :return: the optimizer as a tensor
        """
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def accuracy_function(self):
        """
        :return: the accuracy of the model as a tensor
        """
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                      tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():

    # import MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    batch_num = 2000
    batch_size = 50

    input_image = tf.placeholder(tf.float32, shape=(batch_size, 784))
    input_label = tf.placeholder(tf.float32, shape=(batch_size, 10))

    # init model and tensorflow variables
    model = Model(input_image, input_label)

    # run the model on test data and print the accuracy
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(batch_num):
        x, y = mnist.train.next_batch(batch_size)
        sess.run(model.optimize, feed_dict={model.image: x, model.label: y})

    test_x, test_y = mnist.test.next_batch(batch_size)
    print(sess.run(
        model.accuracy, feed_dict={
            model.image: test_x,
            model.label: test_y
        }))


if __name__ == '__main__':
    main()
