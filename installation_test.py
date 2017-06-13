#import statment below is to remove the warning messages listed in 'https://github.com/tensorflow/tensorflow/issues/7778'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# import tensor flow library

import tensorflow as tf

session = tf.Session()


#verify that we can print a string

#tensor flow has a method called 'constant' which will take any input and converts it to a constant
string_constant = tf.constant("String constant from Tensor flow")
number_constant1 = tf.constant(20)
number_constant2 = tf.constant(30)

print(session.run(string_constant))
print "Sum of 20 and 30 is :", session.run(number_constant1 + number_constant2)
