# coding=utf-8
"""Implementation of MI-FGSM attack."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())

import os
import numpy as np
#import cv2
import pandas as pd
import scipy.stats as st
#from scipy.misc import imread, imsave
from matplotlib.image import imread, imsave
import tensorflow as tf
#print(tf.__version__)
#import tensorflow.compat.v1 as tf
from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate
import time
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
#from nets import inception_v4
#from nets import inception_v3
#from nets import inception_resnet_v2
#from nets import resnet_v2
import random

import tensorflow as tf



assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()



slim = tf.contrib.slim

tf.flags.DEFINE_integer('batch_size', 2, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 12, 'max iteration.')
tf.flags.DEFINE_integer('sample_n', 30, 'sample_n')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')
tf.flags.DEFINE_float('std', 0.05, 'std for gaussian sampling.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

tf.flags.DEFINE_string('checkpoint_path', './models',
                       'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('input_dir', './dev_data',
                       'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './outputs','Output directory with images.')

FLAGS = tf.flags.FLAGS

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt')}


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel = gkern(7, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)
#base_dir="./Outputs_parameters/new/M_NSI_FGSM_Inc_v4/STD/"

def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.JPEG')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    temp_dir=output_dir
    
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(temp_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5)


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

@tf.function
def if_elif(x,y,i,j,grad1,num_classes,logits_v3):
    if tf.math.less_equal(i,tf.constant(2)):
        _,_,_,noise = tf.while_loop(stop1, Grad, [x, y, j, grad1])
    else:
        one_hot = tf.one_hot(y, num_classes)
        cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)
        noise = tf.gradients(cross_entropy, x)[0]
    return noise
def f1(x,y,j,grad1):
    _,_,_,noise=tf.while_loop(stop1, Grad, [x, y, j, grad1])
    return noise
def f2(cross_entropy,x):
    noise=tf.gradients(cross_entropy, x)[0]
    return noise

def graph(x, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    pred = tf.argmax(end_points_v3['Predictions'], 1)

    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred + (1 - first_round) * y
    j=tf.constant(0)
    grad1 = tf.zeros(shape=x.get_shape())

    #noise=if_elif(x,y,i,j,grad1,num_classes,logits_v3)
    one_hot = tf.one_hot(y, num_classes)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)
    noise=tf.cond(i<2,lambda:f1(x,y,j,grad1),lambda:f2(cross_entropy,x))

    
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, y, i, x_max, x_min, noise
def Grad(x, y, j, grad):
	eps = 2.0 * FLAGS.max_epsilon / 255.0
	num_iter = FLAGS.num_iter
	alpha = eps / num_iter
	momentum = FLAGS.momentum
	num_classes = 1001
	std=FLAGS.std
	x=x+tf.random_normal(shape=x.get_shape(),mean=0.0,stddev=std,dtype=tf.float32)
	with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
		logits_v3, end_points_v3 = inception_v3.inception_v3(
			x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

	pred = tf.argmax(end_points_v3['Predictions'], 1)

	one_hot = tf.one_hot(y, num_classes)

	cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)
	noise = tf.gradients(cross_entropy, x)[0]
	noise = grad + tf.sign(noise)
	j = tf.add(j, 1)

	return x, y, j, noise

def stop(x, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)



def stop1(x, y, j, grad):
    num_iter = FLAGS.sample_n
    return tf.less(j, num_iter)

def image_augmentation(x):
    # img, noise
    one = tf.fill([tf.shape(x)[0], 1], 1.)
    zero = tf.fill([tf.shape(x)[0], 1], 0.)
    transforms = tf.concat([one, zero, zero, zero, one, zero, zero, zero], axis=1)
    rands = tf.concat([tf.truncated_normal([tf.shape(x)[0], 6], stddev=0.05), zero, zero], axis=1)
    return images_transform(x, transforms + rands, interpolation='BILINEAR')


def image_rotation(x):
    """ imgs, scale, scale is in radians """
    rands = tf.truncated_normal([tf.shape(x)[0]], stddev=0.05)
    return images_rotate(x, rands, interpolation='BILINEAR')


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret

def main(_):
    
    f2l = load_labels('./dev_data/val_rs.csv')
    eps = 2 * FLAGS.max_epsilon / 255.0

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)
    temp_dir=FLAGS.output_dir
    check_or_create_dir(temp_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape) 
        
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        
        x_adv, _, _, _, _, grad = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        #s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        #s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        # s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
        # s5 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        # s6 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        # s8 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            #s2.restore(sess, model_checkpoint_map['inception_v4'])
            #s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            # s4.restore(sess, model_checkpoint_map['resnet_v2'])
            # s5.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            # s6.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            # s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            # s8.restore(sess, model_checkpoint_map['adv_inception_v3'])
            
            idx = 0
            l2_diff = 0
            since = time.time()
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                
                idx = idx + 1
                print("start the i={} attack".format(idx),flush=True)
                #break
                #########----------------------
                #  check alread exists
                #################################
                #file_path=os.path.join(temp_dir, filenames[0])
                #if os.path.exists(file_path):
                #    continue    
                
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                #print("grad",grad/30)
                save_images(adv_images, filenames, FLAGS.output_dir)
                diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255
                l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))
                
              
            time_elapsed = time.time() - since
            print('{:.2f}'.format(l2_diff * FLAGS.batch_size / 1000))
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()
