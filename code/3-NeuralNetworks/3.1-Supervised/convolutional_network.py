"""
 * @Author: 汤达荣 
 * @Date: 2018-01-12 16:14:00 
 * @Last Modified by:   汤达荣 
 * @Last Modified time: 2018-01-12 16:14:00 
 * @Email: tdr1991@outlook.com 
""" 
#coding:utf-8

import tensorflow as tf

from com.baseNet import BaseNet

# Training Parameters
learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units

def leakyrule(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

def conv_net(x, num_classes):
    name = "covn_net"
    with tf.variable_scope(name):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, 32, 5, strides=(2, 2), padding="same")  
        x = tf.layers.batch_normalization(x)
        x = leakyrule(x)
        x = tf.layers.conv2d(x, 64, 5, strides=(2, 2), padding="same")  
        x = tf.layers.batch_normalization(x)
        x = leakyrule(x)
        x_dim = x.get_shape().as_list()
        print(x_dim)
        x = tf.reshape(x, shape=[-1, x_dim[1] * x_dim[2] * x_dim[3]])
        x_dim = x.get_shape().as_list()
        print(x_dim)
        x = tf.layers.dense(x, 1024)
        x = tf.layers.batch_normalization(x)
        x = leakyrule(x)
        x = tf.layers.dense(x, num_classes)
    return x, name

bn = BaseNet(conv_net, learning_rate, num_steps, batch_size, display_step)
bn.train()

