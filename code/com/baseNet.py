"""
 * @Author: 汤达荣 
 * @Date: 2018-01-12 16:59:18 
 * @Last Modified by:   汤达荣 
 * @Last Modified time: 2018-01-12 16:59:18 
 * @Email: tdr1991@outlook.com 
""" 
#coding:utf-8

import platform
import os

#过滤警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import com.gpuManager as gm

class BaseNet():
    def __init__(self, model, learning_rate=0.1, num_steps=1000, batch_size=128, display_step=100):
        if platform.system().lower() == "windows":
            self.mnist = input_data.read_data_sets("E:\\DataScience\\deeplearning\\data", one_hot=True)
        elif platform.system().lower() == "linux":
            self.mnist = input_data.read_data_sets("/data/tdr/deeplearning/data", one_hot=True)        
        self.learning_rate = learning_rate
        self.num_steps =num_steps
        self.batch_size = batch_size
        self.display_step = display_step 
        self.model = model
        #print(num_input)
    
    def select_device(self):
        #有GPU首选GPU，没有采用CPU
        num_gpus = 0
        sel_device = None
        if gm.check_gpus():
            gum = gm.GPUManager()
            num_gpus = gum.get_gpu_num()
            sel_device = gum.auto_choice()
        else:
            print("不存在GPU，将使用cpu")
            sel_device = tf.device("/cpu:0")
        return sel_device

    def train(self):
        num_input = self.mnist.train.images.shape[1]
        num_classes = self.mnist.train.labels.shape[1]  
        x = tf.placeholder(tf.float32, shape=[None, num_input])
        y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

        #模型的输出
        y, modelScope = self.model(x, num_classes)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        loss_op = tf.reduce_mean(cross_entropy)

        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=modelScope)
        c_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=modelScope)

        with tf.control_dependencies(c_update_ops):
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op, var_list=c_vars)

        prediction = tf.nn.softmax(y)

        correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y_, axis=1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        y_op = tf.argmax(prediction, axis=1)

        init = tf.global_variables_initializer()

        #tensorflow在训练时默认占用所有GPU显存，通过设置 allow_growth，显存分配则按需求增长
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sel_device = self.select_device()

        with sel_device:
            with tf.Session(config=config) as sess:
                sess.run(init)
                for step in range(1, self.num_steps + 1):
                    batch_x, batch_y = self.mnist.train.next_batch(self.batch_size)
                    # Run optimization op (backprop)
                    sess.run(train_op, feed_dict={x: batch_x, y_: batch_y})
                    if step % self.display_step == 0 or step == 1:
                        # Calculate batch loss and accuracy
                        loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_x,
                                                                            y_: batch_y})
                        print("Step " + str(step) + ", Minibatch Loss= " + \
                            "{:.4f}".format(loss) + ", Training Accuracy= " + \
                            "{:.3f}".format(acc))
                print("Optimization Finished!")

                # Calculate accuracy for MNIST test images
                print("Testing Accuracy:", \
                    sess.run(accuracy, feed_dict={x: self.mnist.test.images,
                                                y_: self.mnist.test.labels}))

if __name__ == "__main__":
    test = Train()