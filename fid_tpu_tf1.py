'''
From https://github.com/tsc2017/Frechet-Inception-Distance
Code derived from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

Usage:
    Call get_fid(images1, images2, session=YOUR_SESSION, strategy=YOUR_TPUSTRATEGY)
Args:
    images1, images2: Numpy arrays with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary. 
    dtype of the images is recommended to be np.uint8 to save CPU memory.
Returns:
    Frechet Inception Distance between the two image distributions.
'''

import tensorflow as tf
import os
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
tfgan = tf.contrib.gan
FIRST_RUN=[1]
session=tf.compat.v1.InteractiveSession()
# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 8
INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05_v4.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score_tpu.pb'
# Run images through Inception.
inception_images = [None]
activations1 = [None]
activations2 = [None]
fcd = [None]

def inception_activations(num_splits = 1):
    images = inception_images[0]
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.compat.v1.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    activations = tf.map_fn(
        fn = functools.partial(
             tfgan.eval.run_inception, 
             default_graph_def_fn = functools.partial(
             tfgan.eval.get_graph_def_from_url_tarball, 
             INCEPTION_URL, 
             INCEPTION_FROZEN_GRAPH, 
             os.path.basename(INCEPTION_URL)), 
             output_tensor = 'pool_3:0'),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations

activations =[None]

def get_inception_activations(inps, session=None, strategy=None):
    if FIRST_RUN[0]:
        with session.graph.as_default():
            inception_images[0] = tf.compat.v1.placeholder(tf.float32, [None, 3, None, None], name = 'inception_images')
            activations1[0] = tf.compat.v1.placeholder(tf.float32, [None, None], name = 'activations1')
            activations2[0] = tf.compat.v1.placeholder(tf.float32, [None, None], name = 'activations2')
            fcd[0] = tfgan.eval.frechet_classifier_distance_from_activations(activations1[0], activations2[0])
            print('Running Inception for the first time, compiling...')
            activations[0]=strategy.experimental_run(inception_activations).values[0]
            FIRST_RUN[0]=0
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    act = np.zeros([inps.shape[0], 2048], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] / 255. * 2 - 1
        act[i * BATCH_SIZE : i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(activations[0], feed_dict = {inception_images[0]: inp})
    return act

def activations2distance(act1, act2, session=None):
    return session.run(fcd[0], feed_dict = {activations1[0]: act1, activations2[0]: act2})
        
def get_fid(images1, images2, session=None, strategy=None):
    assert(type(images1) == np.ndarray)
    assert(len(images1.shape) == 4)
    assert(images1.shape[1] == 3)
    assert(np.min(images1[0]) >= 0 and np.max(images1[0]) > 10), 'Image values should be in the range [0, 255]'
    assert(type(images2) == np.ndarray)
    assert(len(images2.shape) == 4)
    assert(images2.shape[1] == 3)
    assert(np.min(images2[0]) >= 0 and np.max(images2[0]) > 10), 'Image values should be in the range [0, 255]'
    assert(images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
    print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(images1, session, strategy)
    act2 = get_inception_activations(images2, session, strategy)
    fid = activations2distance(act1, act2, session)
    print('FID calculation time: %f s' % (time.time() - start_time))
    return fid
