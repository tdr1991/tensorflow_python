#coding:utf-8

import tensorflow as tf

from com.baseNet import BaseNet

#超参数
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

#网络参数
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons

#定义模型
def neural_net(x, num_classes):
    name = "neural_net"
    with tf.variable_scope(name):
        x = tf.layers.dense(x, n_hidden_1)
        x = tf.layers.dense(x, n_hidden_2)
        x = tf.layers.dense(x, num_classes)
    return x, name

bn = BaseNet(neural_net, learning_rate, num_steps, batch_size, display_step)
bn.train()
"""
#定义输入输出占位符
x = tf.placeholder(tf.float32, shape=[None, num_input])
y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

#模型的输出
y = neural_net(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
loss_op = tf.reduce_mean(cross_entropy)

c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="neural_net")
c_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="neural_net")

with tf.control_dependencies(c_update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op, var_list=c_vars)

prediction = tf.nn.softmax(y)

correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y_, axis=1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

y_op = tf.argmax(prediction, axis=1)

init = tf.global_variables_initializer()

#tensorflow在训练时默认占用所有GPU显存，通过设置 allow_growth，显存分配则按需求增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

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

with sel_device:
    with tf.Session(config=config) as sess:
        sess.run(init)
        for step in range(1, num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={x: batch_x, y_: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_x,
                                                                    y_: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                    "{:.3f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))
#print(num_input, num_classes)
#print(gm.check_gpus())
"""