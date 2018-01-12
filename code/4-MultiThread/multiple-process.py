#coding:utf-8

import os
import sys

#过滤警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

curDir = os.getcwd()
print(curDir)
#pkPath = curDir + os.sep + "code"
print(sys.path)
#sys.path.append(pkPath)
print(sys.path)

import com.gpuManager as gm

 