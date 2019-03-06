import tensorflow as tf
import numpy as np
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BATCH_SIZE = 50
WINDOW_SIZE = 20


def vocab_int_mapping_dict(words):
    words = set(words)
    vocab = dict()
    for i, word in enumerate(words):
        vocab[word] = i
        # vocab[i] = word

    return vocab


def read(train_file, test_file):
    train_data = None
    test_data = None

    with open(train_file) as file:
        train_data = file.read()
    train_data = train_data.split()

    with open(test_file) as file:
        test_data = file.read()
    test_data = test_data.split()

    vocab = vocab_int_mapping_dict(train_data)

    train_x, train_y = [], []
    for word_index in range(0, len(train_data)):
        train_x.append(vocab[train_data[word_index]])
        label_index = word_index + 1
        if label_index < len(train_data):
            train_y.append(vocab[train_data[label_index]])
        else:
            train_y.append(vocab["STOP"])

    test_x, test_y = [], []
    for word_index in range(0, len(test_data)):
        test_x.append(vocab[test_data[word_index]])
        label_index = word_index + 1
        if label_index < len(test_data):
            test_y.append(vocab[test_data[label_index]])
        else:
            test_y.append(vocab["STOP"])

    return train_x, train_y, test_x, test_y, vocab


class Model:
    def __init__(self, inputs, labels, keep_prob, vocab_size):
        self.learning_rate = 1e-3
        self.rnn_size = 300
        self.embed_size = 150

        # Input tensors, DO NOT CHANGE
        self.inputs = inputs
        self.labels = labels
        self.keep_prob = keep_prob

        # DO NOT CHANGE
        self.vocab_size = vocab_size
        self.prediction = self.forward_pass()  # Logits for word predictions
        self.loss = self.loss_function()  # The average loss of the batch
        self.optimize = self.optimizer()  # An optimizer (e.g. ADAM)
        self.perplexity = self.perplexity_function()  # The perplexity of the model, Tensor of size 1

    def forward_pass(self):
        """
        :return: logits: The prediction logits as a tensor
        """
        e = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size], stddev=0.1))
        embeddings = tf.nn.embedding_lookup(e, self.inputs)
        embeddings = tf.nn.dropout(embeddings, self.keep_prob)

        rnn = tf.contrib.rnn.LSTMCell(self.rnn_size)
        outputs, nextState = tf.nn.dynamic_rnn(rnn, embeddings, dtype=tf.float32)

        W = tf.Variable(tf.truncated_normal([self.rnn_size, self.rnn_size // 2], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([self.rnn_size // 2], stddev=0.1))

        o1 = tf.nn.elu(tf.tensordot(outputs, W, [[2], [0]]) + b)

        W1 = tf.Variable(tf.truncated_normal([self.rnn_size // 2, self.vocab_size], stddev=0.1))
        b1 = tf.Variable(tf.truncated_normal([self.vocab_size], stddev=0.1))

        o2 = tf.tensordot(o1, W1, [[2], [0]]) + b1

        return o2

    def optimizer(self):
        """
        :return: the optimizer as a tensor
        """
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def loss_function(self):
        """
        :return: the loss of the model as a tensor of size 1
        """
        return tf.contrib.seq2seq.sequence_loss(self.prediction, self.labels, tf.ones_like(self.labels, dtype=tf.float32))

    def perplexity_function(self):
        """
        :return: the perplexity of the model as a tensor of size 1
        """
        return tf.exp(self.loss)


def main():
    train_file = "train.txt"
    dev_file = "dev.txt"
    train_x, train_y, test_x, test_y, vocab_map = read(train_file, dev_file)

    inputs = tf.placeholder(tf.int32, shape=[None, WINDOW_SIZE])
    output = tf.placeholder(tf.int32, shape=[None, WINDOW_SIZE])

    batched_x = []
    batched_y = []
    sections_size = math.floor((len(train_x)) / BATCH_SIZE)

    for b in range(0, BATCH_SIZE):
        to_add = []
        to_add_y = []
        for i in range(sections_size):
            to_add.append(train_x[b*sections_size + i])
            to_add_y.append(train_y[b*sections_size + i])
        batched_x.append(to_add)
        batched_y.append(to_add_y)

    vocab_sz = len(vocab_map)  # // 2
    keep_prob = 0.9
    model = Model(inputs, output, keep_prob, vocab_sz)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    batched_x = np.array(batched_x)
    batched_y = np.array(batched_y)

    for i in range(0, sections_size, WINDOW_SIZE):
        if (i + WINDOW_SIZE < sections_size):
            batched_x_slice = np.take(batched_x, range(i, i + WINDOW_SIZE), axis=1)
            batched_y_slice = np.take(batched_y, range(i, i + WINDOW_SIZE), axis=1)

            sess.run(model.optimize, feed_dict={model.inputs: batched_x_slice, model.labels: batched_y_slice})
            print("progress", i/len(batched_x[0]), "perplexity", sess.run(
                model.perplexity, feed_dict={model.inputs: batched_x_slice, model.labels: batched_y_slice}))

    test_x = np.array([test_x])
    test_y = np.array([test_y])
    perplexities = []
    for i in range(0, len(test_x[0]), WINDOW_SIZE):
        if (i + WINDOW_SIZE < len(test_x[0])):
            batched_x_slice = np.take(test_x, range(i, i + WINDOW_SIZE), axis=1)
            batched_y_slice = np.take(test_y, range(i, i + WINDOW_SIZE), axis=1)
            perplexities.append(sess.run(
                model.perplexity, feed_dict={model.inputs: batched_x_slice, model.labels: batched_y_slice}))
    print("test perplexity", np.mean(np.array(perplexities)))


if __name__ == '__main__':
    main()
