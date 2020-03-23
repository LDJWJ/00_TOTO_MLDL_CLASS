import tensorflow as tf

print(tf.__version__)
hello = tf.constant("Hello, Tensorflow")  # 상수값 저장
print(hello)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print( sess.run(hello) )
sess.close()