import tensorflow as tf
import numpy as np
import os
import sys
import glob
import cv2

num_models = 5
sex_path = "./savedModel/s/"
age_path = "./savedModel/a/"

class Model:
    """
    이전에 학습된 모델과 같은 계층 구조와 변수들을 가진 클래스
    이 클래스로 그래프를 만들고 거기에 각 변수를 읽어와서 대입함
    """
    def __init__(self, sess, name):
        self.name = name
        self._build_net()
        self.saver = tf.Saver()
        self.sess = sess

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 128, 128, 1], name = 'X')
            self.Y = tf.placeholder(tf.float32, [None, 2], name = 'Y')

            # Convolution Layer 1 and Pooling Layer 1
            conv1 = tf.layers.conv2d(
                inputs = self.X,
                filters = 32,
                kernel_size = [5, 5],
                padding = 'SAME',
                activation = tf.nn.relu,
                name = 'conv1')
            pool1 = tf.layers.max_pooling2d(
                inputs = conv1,
                pool_size = [2, 2],
                strides = 2,
                padding = 'SAME')
            dropout1 = tf.layers.dropout(
                inputs = pool1,
                rate = 0.7,
                training = self.training)

            # Convolution Layer 2 and Pooling Layer 2
            conv2 = tf.layers.conv2d(
                inputs = dropout1,
                filters = 64,
                kernel_size = [3, 3],
                padding = 'SAME',
                activation = tf.nn.relu,
                name = 'conv2')
            pool2 = tf.layers.max_pooling2d(
                inputs = conv2,
                pool_size = [2, 2],
                strides = 2,
                padding = 'SAME')
            dropout2 = tf.layers.dropout(
                inputs = pool2,
                rate = 0.7,
                training = self.training)

            # Dense Layer
            flat = tf.reshape(dropout2, [-1, 32 * 32 * 64])
            dense3 = tf.layers.dense(
                inputs = flat,
                units = 512,
                activation = tf.nn.relu,
                name = 'dense')
            dropout3 = tf.layers.dropout(
                inputs = dense3,
                rate = 0.5,
                training = self.training)

            # Logits Layer (FC 512 inputs -> 2 outputs)
            self.logits = tf.layers.dense(
                inputs = dropout3,
                units = 2,
                name = 'output')

        # Define cost, optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def predict(self, x_test, training = False):
        return self.sess.run(
            self.logits,
            feed_dict = {self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training = False):
        return self.sess.run(
            self.acc,
            feed_dict = {self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training = True):
        return self.sess.run(
            [self.cost, self.optimizer],
            feed_dict = {self.X: x_data, self.Y: y_data, self.training: training})

    def save(self, path):
        self.saver.save(sess=self.sess, save_path=path+self.name)

    def load(self, path):
        self.saver.restore(self.sess, path+self.name)


def get_path(path, num):
    dirt = path+str(num)+'/'
    if not os.path.exists(dirt):
        os.makedirs(dirt)
    return dirt

def get_data(path):
    d = []
    os.chdir(path)
    for image in glob.glob("*.jpg"):
        tmp = []
        img = cv2.imread(image)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(128):
            for j in range(128):
                tmp.append(gray_n[i, j]) #변경된 부분
        d.append(tmp)
    return d

def predict_sex(path, models):
    data = get_data(path)
    size = len(data)
    predictions = np.zeros(size * 2).reshape(size, 2)
    for m_idx, m in enumerate(models):
        print(m_idx, 'Accuracy:', m.get_accuracy(
            test_data, test_label))
        p = m.predict(test_data)
        predictions += p
    pred = sess.run(tf.argmax(predictions, 1))
    return pred

def predict_sex(path, models):
    data = get_data(path)
    size = len(data)
    predictions = np.zeros(size * 2).reshape(size, 2)
    for m_idx, m in enumerate(models):
        print(m_idx, 'Accuracy:', m.get_accuracy(
            test_data, test_label))
        p = m.predict(test_data)
        predictions += p
    pred = sess.run(tf.argmax(predictions, 1))
    return pred

sess = tf.Session()

sex_models = []
for m in range(num_models):
    sex_models.append(Model.Model(sess, "model" + str(m)))
    save_dir = get_path(sex_path, m)
    sex_models[m].load(save_dir)

age_models = []
for m in range(num_models):
    age_models.append(Model.Model(sess, "model" + str(m)))
    save_dir = get_path(age_path, m)
    age_models[m].load(save_dir)

while True:
    line = sys.stdin.readline()
    pred = predict_sex(line, sex_models)
    for p in pred:
        print(p)
