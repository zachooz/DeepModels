import numpy as np
import os
import gzip
from collections import namedtuple

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model:
    """
        Arguments:
        train_images - NumPy array of training images
        train_labels - NumPy array of labels
    """
    def __init__(self, train_images, train_labels):
        input_size, num_classes, batchSz, learning_rate = 784, 10, 1, 0.5
        self.train_images = train_images
        self.train_labels = train_labels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batchSz = batchSz
        # sets up weights and biases...
        self.W = np.random.normal(0, .01, (input_size, num_classes))
        self.b = np.random.normal(0, .01, num_classes)
        self.order = np.arange(len(self.train_labels))
        self.current_image_index = 0
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.order)

    def run(self):
        """
        Does the forward pass, loss calculation, and back propagation
        for this model FOR ONE STEP
        """
        data = self.train_images
        labels = self.train_labels

        # Check to shuffle data
        if self.current_image_index > len(data) - 1:
            self.current_image_index = 0
            self.shuffle()
            self.train_images = self.train_images[self.order]
            self.train_labels = self.train_labels[self.order]

        index = self.current_image_index

        x = data[index:index+self.batchSz]
        y = labels[index:index+self.batchSz]

        output = np.matmul(x, self.W) + self.b
        # print("o", output)
        softmaxed = self.softmax(output)
        # loss = -1 * np.sum(np.log(softmaxed))
        gradient = np.zeros((self.batchSz, self.num_classes))

        # backward pass
        for b in range(self.batchSz):
            for i in range(self.num_classes):
                if i == y[b]:
                    gradient[b][i] = softmaxed[b][i] - 1
                else:
                    gradient[b][i] = softmaxed[b][i]

        g_mean = np.mean(gradient, axis=0)
        self.b = self.b - self.learning_rate * g_mean

        gradient = np.matmul(np.transpose(x), gradient)
        self.W = self.W - self.learning_rate * gradient

        self.current_image_index += self.batchSz

    def softmax(self, x):
        """ apply softmax to an array
        param:
            x: np array
        return:
            array with softmax applied elementwise.
        """
        e = np.exp(x - np.max(x, axis=1)[:, None])  # subtract max to avoid large exponent values e = np.exp(x - np.max(x))
        return e / np.sum(e, axis=1)[:, None]

    def accuracy_function(self, test_images, test_labels):
        """
        Calculates the accuracy of the model against test images and labels

        DO NOT EDIT
        Arguments
        test_images: a normalized NumPy array
        test_labels: a NumPy array of ints
        """
        scores = np.dot(test_images, self.W) + self.b
        predicted_classes = np.argmax(scores, axis=1)
        return np.mean(predicted_classes == test_labels)


def main():
    # TO-DO: import MNIST test data
    # DATA IMPORT FROM CS 142
    Dataset = namedtuple('Dataset', ['inputs', 'labels'])
    data_train = None
    data_test = None

    with open("data/train-images-idx3-ubyte.gz", 'rb') as f1, open("data/train-labels-idx1-ubyte.gz", 'rb') as f2:
        buf1 = gzip.GzipFile(fileobj=f1).read(16 + 10000 * 28 * 28)
        buf2 = gzip.GzipFile(fileobj=f2).read(8 + 10000)
        inputs = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(10000, 28 * 28)
        inputs = np.true_divide(inputs, 255)
        labels = np.frombuffer(buf2, dtype='uint8', offset=8)
        data_train = Dataset(inputs, labels)
    with open("data/t10k-images-idx3-ubyte.gz", 'rb') as f1, open("data/t10k-labels-idx1-ubyte.gz", 'rb') as f2:
        buf1 = gzip.GzipFile(fileobj=f1).read(16 + 10000 * 28 * 28)
        buf2 = gzip.GzipFile(fileobj=f2).read(8 + 10000)
        inputs = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(10000, 28 * 28)
        inputs = np.true_divide(inputs, 255)
        labels = np.frombuffer(buf2, dtype='uint8', offset=8)
        data_test = Dataset(inputs, labels)

    model = Model(data_train.inputs, data_train.labels)
    for i in range(10000):
        model.run()
    print("accuracy", model.accuracy_function(data_test.inputs, data_test.labels))


if __name__ == '__main__':
    main()
