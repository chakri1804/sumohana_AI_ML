import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/',one_hot=True)
hl_1 = 500
hl_2 = 500
hl_3 = 500
classes = 10
batch_size = 1000
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_net(data):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784,hl_1])),
                     'biases':tf.Variable(tf.random_normal([hl_1]))}
    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([hl_1,hl_2])),
                     'biases':tf.Variable(tf.random_normal([hl_2]))}
    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([hl_2,hl_3])),
                     'biases':tf.Variable(tf.random_normal([hl_3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([hl_3,classes])),
                     'biases':tf.Variable(tf.random_normal([classes]))}

    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)
    out = tf.matmul(l3,output_layer['weights'])+output_layer['biases']
    return out

def train(x):
    prediction = neural_net(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 50

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print(epoch,'/',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train(x)
