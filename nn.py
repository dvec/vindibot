from sys import stdout

import tensorflow as tf


class NN:
    _DROPOUT = 0.02

    def __init__(self):
        self._x = tf.placeholder(dtype=tf.float32, shape=[1200, 820], name='x-input')
        self._y = tf.placeholder(dtype=tf.float32, shape=[1200, 5], name='y-input')

        l1 = tf.layers.dropout(tf.layers.dense(self._x, 820, activation=tf.tanh), rate=self._DROPOUT)
        l2 = tf.layers.dropout(tf.layers.dense(l1, 820, activation=tf.tanh), rate=self._DROPOUT)
        l3 = tf.layers.dropout(tf.layers.dense(l2, 100, activation=tf.sigmoid), rate=self._DROPOUT)

        self._p = tf.layers.dropout(tf.layers.dense(l3, 5, activation=tf.tanh), rate=self._DROPOUT, name='P1')

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def train(self, g, epoch=30, learning_rate=0.01, save=100, log=True, verbose=1):
        cost = tf.reduce_mean((self._p - self._y) ** 2)
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        summary_writer = tf.summary.FileWriter('/tmp/tensorboard/', graph=tf.get_default_graph())

        for i in range(epoch):
            for x, y in g:
                _, c = self._sess.run([train, cost], feed_dict={self._x: x, self._y: y})
                if save and i % save == 0:
                    self.save(step=i)

                summary_writer.add_summary(c)
                if verbose:
                    stdout.write('\rLoss: {}'.format(c))

    def run(self, x, verbose=1):
        return self._sess.run(self._p, feed_dict={self._x: x})

    def save(self, step=None, file='./models/model'):
        saver = tf.train.Saver()
        saver.save(self._sess, file, global_step=step)

    def load(self, checkpoint_dir='./models/'):
        saver = tf.train.Saver()
        saver.restore(self._sess, tf.train.latest_checkpoint(checkpoint_dir))
