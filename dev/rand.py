import tensorflow as tf

a = tf.random.uniform((2,10,64))

trans = tf.transpose(a, [1,0,2])
stack = tf.unstack(trans, axis=0)

print(stack)
