"""
helper functions and network structures
"""
#pylint: disable=C0411, C0103, C0301, C0121, E1101
from __future__ import division
import tensorflow as tf
import os
import numpy as np
import scipy.misc
from glob import glob
WEIGHTS_INIT_STDEV = .1

H = 16
#from odl_recon import recon, recon45
T = 2
def get_filter(f_len):
    """
    The filter of the filtered back projection (FBP) operator.
    Args:
        f_len: filter length, integer.
    Return:
        Tensorflow varibale of shape [1, f_len, 1, 1].
    """
    f = np.zeros([1, f_len, 1, 1])
    mid = int((f_len-1)/2)
    for i in range(f_len):
        x = i - mid
        if x % 2 == 1:
            f[0, i, 0, 0] = -1 / np.square(x * np.pi)
        f[0, mid, 0, 0] = 1 / 4
        fil = tf.get_variable('filter', [1, f_len, 1, 1], tf.float32, initializer=tf.constant_initializer(f), trainable=False)
        return fil

def get_f(f_len):
    """
    The filter of the filtered back projection (FBP) operator.
    Args:
        f_len: filter length, integer.
    Return:
        numpy array of length f_len
    """
    f = np.zeros([f_len])
    mid = int((f_len - 1) / 2)
    for i in range(f_len):
        x = i-mid
        if x%2 == 1:
            f[i] = -1 / np.square(x * np.pi)
    f[mid] = 1 / 4
    return f

def Prelu(net, name='prelu'):
    """
    Parametric Rectified Linear Units
    """
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', net.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    return tf.maximum(0.0, net) + alphas * tf.minimum(0.0, net)

def load_img(name, rot):
    """
    load .npy image file with possible rotation.
    """
    if 'rot1' in rot:
        resu = np.rot90(np.load(name), 1)
    elif 'rot2' in rot:
        resu = np.rot90(np.load(name), 2)
    elif 'rot3' in rot:
        resu = np.rot90(np.load(name), 3)
    else:
        resu = np.load(name)
    return resu

def list_files(in_path, name='*.jpg'):
    """
    list files in a folder.
    """
    return glob(os.path.join(in_path, name))

def get_image(img_path, img_size=512, is_crop=False):
    """
    Read image from a .jpg or .png file, operating crop or resize to a given img_size.
    Args:
        img_path: The image file path.
        img_size: The size of the output image.
        is_crop: If the output image has been cropped.
    Returns:
        numpy array of size [img_size, img_size]
    """
    img = scipy.misc.imread(img_path, mode='L').astype(np.float32)
    #img[img<0.9*(img[10, 256])] = 0.9 * img[10, 256]
    m, n = img.shape
    if is_crop == True:
        cm = int(round(m/2))
        cn = int(round(n/2))
        img = img[cm-int(img_size//2):cm+int(img_size//2), cn-int(img_size//2):cn+int(img_size//2)]
    else:
        img = scipy.misc.imresize(img, [img_size, img_size])
    img = img/255*2000.0
    return img

def save_image(out_path, img):
    """
    save image
    """
    scipy.misc.imsave(out_path, img)
    print('saved')

def U_net(inputs, inputs2, train=True, reuse=False, name='U_net'):
    conv_initializer = tf.contrib.layers.savier_initializer()
    with tf.variable_scope(name) as scope:
        if reuse == True:
            scope.reuse_variables()
        x = tf.concat([inputs-inputs2, inputs2], 3)
        layers = []
        #net = inputs2
        for n in [64, 128, 256, 512]:
            x = tf.layers.conv2d(x, n, 3, padding='SAME', activation=tf.nn.relu, kernel_initializer=conv_initializer)
            x = tf.layers.batch_normalization(x, scale=False, training=train)

            x = tf.layers.conv2d(x, n, 3, padding='SAME', activation=tf.nn.relu, kernel_initializer=conv_initializer)
            x = tf.layers.batch_normalization(x, scale=False, training=train)
            layers.append(x)
            x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')

        x = tf.layers.conv2d(x, 1024, 3, padding='SAME', activation=tf.nn.relu, kernel_initializer=conv_initializer)
        x = tf.layers.batch_normalization(x, scale=False, training=train)
        x = tf.layers.conv2d(x, 512, 3, padding='SAME', activation=None, kernel_initializer=conv_initializer)
        x = tf.layers.batch_normalization(x, scale=False, training=train)
        x = concat(layers[-1], upscale(x))
        pos = -1

        for n in [512, 256, 128]:
            pos -= 1
            x = tf.layers.conv2d(x, n, 3, activation=tf.nn.relu, kernel_initializer=conv_initializer)
            x = tf.layers.batch_normalization(x, scale=False, training=train)

            x = tf.layers.conv2d(x, int(n//2), 3, activation=tf.nn.relu, kernel_initializer=conv_initializer)
            x = tf.layers.batch_normalization(x, scale=False, training=train)
            x = concat(layers[pos], upscale(x))

        x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu, kernel_initializer=conv_initializer)
        x = tf.layers.batch_normalization(x, scale=False, training=train)

        x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu, kernel_initializer=conv_initializer)
        x = tf.layers.batch_normalization(x, scale=False, training=train)

        x = tf.layers.conv2d(x, 1, 1, activation=None, kernel_initializer=conv_initializer)

    return inputs2+x

def int_U_net(inputs, times=4, train=True, reuse=False, name='inp_U_net'):
    conv_initializer = tf.contrib.layers.savier_initializer()
    with tf.variable_scope(name) as scope:
        if reuse == True:
            scope.reuse_variables()
        #net = tf.concat([inputs-inputs2, inputs2],3)
        net = inter_up(inputs, times=times)
        x = net
        layers = []
        #net = inputs2
        for n in [64, 128, 256, 512]:
            x = tf.layers.conv2d(x, n, 3, padding='SAME', activation=tf.nn.relu, kernel_initializer=conv_initializer)
            x = tf.layers.batch_normalization(x, scale=False, training=train)

            x = tf.layers.conv2d(x, n, 3, padding='SAME', activation=tf.nn.relu, kernel_initializer=conv_initializer)
            x = tf.layers.batch_normalization(x, scale=False, training=train)
            layers.append(x)
            x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')

        x = tf.layers.conv2d(x, 1024, 3, padding='SAME', activation=tf.nn.relu, kernel_initializer=conv_initializer)
        x = tf.layers.batch_normalization(x, scale=False, training=train)
        x = tf.layers.conv2d(x, 512, 3, padding='SAME', activation=None, kernel_initializer=conv_initializer)
        x = tf.layers.batch_normalization(x, scale=False, training=train)
        x = concat(layers[-1], upscale(x))
        pos = -1

        for n in [512, 256, 128]:
            pos -= 1
            x = tf.layers.conv2d(x, n, 3, activation=tf.nn.relu, kernel_initializer=conv_initializer)
            x = tf.layers.batch_normalization(x, scale=False, training=train)

            x = tf.layers.conv2d(x, int(n//2), 3, activation=tf.nn.relu, kernel_initializer=conv_initializer)
            x = tf.layers.batch_normalization(x, scale=False, training=train)
            x = concat(layers[pos], upscale(x))

        x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu, kernel_initializer=conv_initializer)
        x = tf.layers.batch_normalization(x, scale=False, training=train)

        x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu, kernel_initializer=conv_initializer)
        x = tf.layers.batch_normalization(x, scale=False, training=train)

        x = tf.layers.conv2d(x, 1, 1, activation=None, kernel_initializer=conv_initializer)

    return net+x

def upscale(inputs):
    shape = inputs.get_shape()
    size = [2*int(s) for s in shape[1:3]]
    out = tf.image.resize_nearest_neighbor(inputs, size)
    return out
def inter_up(inputs, times=4):
    shape = inputs.get_shape()
    h, w = [int(s) for s in shape[1: 3]]
    h = times*h
    output = tf.image.resize_images(inputs, [h, w], method=tf.image.ResizeMethod.BICUBIC)
    return output

def concat(inputs1, inputs2):
    """
    concatenation operation of 4 dimension tensor object
    """
    return tf.concat(axis=3, values=[inputs1, inputs2])


def total_variation_loss(x):
    """
    tv loss of an numpy array.
    """
    nrows, ncols = x.shape
    a = np.square(x[:nrows-1, :ncols-1] - x[1:, :ncols-1])
    b = np.square(x[:nrows-1, :ncols-1] - x[:nrows-1, 1:])
    return np.sum((a+b)**1.25)

def tv_loss(preds, tv_weight=1):
    """
    tv loss of an 4 dimension tensor array.
    """
    batch_shape = [i.value for i in preds.get_shape()]
    s = batch_shape[0]*batch_shape[1]*(batch_shape[2]-1)*batch_shape[3]
    y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :batch_shape[1]-1, :, :])
    x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :batch_shape[2]-1, :])
    loss = tv_weight*2*(x_tv + y_tv)/s
    return loss
