'''
IMPORTANT: DO NOT modify the existing stencil code or else it may not be compatible with the Autograder
If you find any problems, report to TA's.
'''
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
import tensorflow.contrib.gan as gan
import math

layers = tf.layers
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_dir = './images_in'
out_dir = './images_out'
batch_size = 128
thread_count = 2
num_epochs = 10
num_updates = 2
save_frequency = 100
log_frequency = 10
z_dim = 100


# Numerically stable logarithm function
def log(x):
    return tf.log(tf.maximum(x, 1e-5))


def conv_same_out_size(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class Model:
    def __init__(self, image_batch, g_input_z):
        self.image_batch = image_batch
        self.g_input_z = g_input_z
        self.learning_rate = 0.0002
        self.momentum = 0.5
        self.output_height = 64
        self.output_width = 64

        # YOUR CODE GOES HERE
        # Finish setting up the TF graph:
        #  - Build the generator graph
        #  - Build the discriminator graph with 2 inputs, one for real image input, one for fake
        #  - You might want to create helper function(s) to help accomplish the task
        self.g_output = self.generator(g_input_z)

        with tf.variable_scope("") as scope:
            self.logits_real = self.discriminator(image_batch)
            scope.reuse_variables()
            self.logits_fake = self.discriminator(self.g_output)

        self.D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        self.G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        # Declare losses, optimizers(trainers) and fid for evaluation
        self.g_loss = self.g_loss_function()
        self.d_loss = self.d_loss_function()
        self.g_train = self.g_trainer()
        self.d_train = self.d_trainer()
        self.fid = self.fid_function()

    # Training loss for Generator
    def g_loss_function(self):
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.logits_fake), logits=self.logits_fake))
        return g_loss

    # Training loss for Discriminator
    def d_loss_function(self):
        d_loss = (
                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.logits_real), logits=self.logits_real))
                    + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.logits_fake), logits=self.logits_fake))
                 )
        return d_loss

    # Optimizer/Trainer for Generator
    def g_trainer(self):
        g_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.momentum).minimize(self.g_loss, var_list=self.G_vars)
        return g_train

    # Optimizer/Trainer for Discriminator
    def d_trainer(self):
        d_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.momentum).minimize(self.d_loss, var_list=self.D_vars)
        return d_train

    def discriminator(self, x):
        with tf.variable_scope("discriminator"):
            o1 = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
            o1 = tf.nn.leaky_relu(tf.layers.batch_normalization(o1))
            o2 = tf.layers.conv2d(o1, 128, 5, strides=2, padding='same')
            o2 = tf.nn.leaky_relu(tf.layers.batch_normalization(o2))
            o3 = tf.layers.conv2d(o2, 256, 5, strides=2, padding='same')
            o3 = tf.nn.leaky_relu(tf.layers.batch_normalization(o3))
            o4 = tf.layers.conv2d(o3, 512, 5, strides=2, padding='same')
            o4 = tf.layers.flatten(o4)
            logits = tf.layers.dense(o4, 1, activation=tf.nn.leaky_relu)
            return logits

    def generator(self, z):
        with tf.variable_scope("generator"):
            # TODO: implement architecture
            d1 = tf.layers.dense(z, 8192)
            o1 = tf.reshape(d1, shape=[-1, 4, 4, 512])
            # o1 = tf.layers.conv2d_transpose(r_z, 512, kernel_size=(2, 2), strides=(2, 2))
            o2 = tf.layers.conv2d_transpose(o1, 256, 5, strides=2, padding='same')
            o2 = tf.nn.relu(tf.layers.batch_normalization(o2))
            o3 = tf.layers.conv2d_transpose(o2, 128, 5, strides=2, padding='same')
            o3 = tf.nn.relu(tf.layers.batch_normalization(o3))
            o4 = tf.layers.conv2d_transpose(o3, 64, 5, strides=2, padding='same')
            o4 = tf.nn.relu(tf.layers.batch_normalization(o4))
            o5 = tf.tanh(tf.layers.conv2d_transpose(o4, 3, 5, strides=2, padding='same'))
            return o5

    # For evaluating the quality of generated images
    # Smaller is better
    def fid_function(self):
        INCEPTION_IMAGE_SIZE = (299, 299)
        real_resized = tf.image.resize_images(self.image_batch, INCEPTION_IMAGE_SIZE)
        fake_resized = tf.image.resize_images(self.g_output, INCEPTION_IMAGE_SIZE)
        return gan.eval.frechet_classifier_distance(real_resized, fake_resized, gan.eval.run_inception)


def sample_noise(batch_size, dim):
    return np.random.uniform(-1, 1, (batch_size, dim))


# Sets up tensorflow graph to load images
def load_image_batch(dirname, batch_size=128, shuffle_buffer_size=250000, n_threads=2):

    # Function used to load and pre-process image files
    def load_and_process_image(filename):
        # Load image
        image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
        # Convert image to normalized float (0, 1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Rescale data to range (-1, 1)
        image = (image - 0.5) * 2
        return image

    # List filenames
    dir_path = dirname + '/*.jpg'
    dataset = tf.data.Dataset.list_files(dir_path)
    # Shuffle order
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    # Load and process images (in parallel)
    dataset = dataset.map(map_func=load_and_process_image, num_parallel_calls=n_threads)
    # Create batch, dropping the final one which has less than
    #    batch_size elements
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size=batch_size))
    # Prefetch the next batch while the GPU is training
    dataset = dataset.prefetch(1)

    return dataset.make_initializable_iterator()


dataset_iterator = load_image_batch(img_dir,
                                    batch_size=batch_size,
                                    n_threads=thread_count)

image_batch = dataset_iterator.get_next()

z = tf.placeholder(tf.float32, [None, 100], name='z')
model = Model(tf.placeholder(tf.float32, [None, 64, 64, 3]), z)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


def load_last_checkpoint():
    saver.restore(sess, tf.train.latest_checkpoint('./'))


def train():
    # Training loop
    for epoch in range(0, num_epochs):
        # Shuffle data
        sess.run(dataset_iterator.initializer)

        # Loop over our data until we run out
        iteration = 0
        try:
            while True:
                minibatch = sess.run(image_batch)
                loss_d, op_d = sess.run([model.d_loss, model.d_train],
                                        feed_dict={
                                            model.image_batch: minibatch,
                                            model.g_input_z: sample_noise(batch_size, z_dim)
                                            })
                loss_g, op_g = None, None
                for i in range(num_updates):
                    loss_g, op_g = sess.run([model.g_loss, model.g_train], feed_dict={model.g_input_z: sample_noise(batch_size, z_dim)})

                # Print losses
                if iteration % log_frequency == 0:
                    print('Iteration %d: Gen loss = %g | Discrim loss = %g' % (iteration, loss_g, loss_d))

                # Save
                if iteration % save_frequency == 0:
                    saver.save(sess, './dcgan_saved_model')
                iteration += 1
        except tf.errors.OutOfRangeError:
            # Triggered when the iterator runs out of data
            pass

        # Save at the end of the epoch, too
        saver.save(sess, './dcgan_saved_model')

        # Also, print the inception distance
        sess.run(dataset_iterator.initializer)

        minibatch = sess.run(image_batch)

        fid_ = sess.run(model.fid,
                        feed_dict={
                            model.image_batch: minibatch,
                            model.g_input_z: sample_noise(batch_size, 100)
                            })  # Use sess.run to get the inception distance value defined above
        print('**** INCEPTION DISTANCE: %g ****' % fid_)


# Test the model by generating some samples from random latent codes
def test():
    gen_img_batch = sess.run(model.g_output, feed_dict={model.g_input_z: sample_noise(batch_size, 100)})

    gen_img_batch = ((gen_img_batch / 2) - 0.5) * 255
    # Convert to uint8
    gen_img_batch = gen_img_batch.astype(np.uint8)
    # Save images to disk
    for i in range(0, batch_size):
        img_i = gen_img_batch[i]
        s = out_dir+'/'+str(i)+'.png'
        imsave(s, img_i)
