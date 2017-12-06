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
        self.saver = tf.train.Saver()
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

    def predict(self, x_test, training = False):
        return self.sess.run(
            self.logits,
            feed_dict = {self.X: x_test, self.training: training})

    def save(self, path):
        self.saver.save(sess=self.sess, save_path=path+self.name)

    def load(self, path):
        self.saver.restore(self.sess, path+self.name)

class AModel:
    """
    나이대 예측 모델
    """
    def __init__(self, sess, name):
        self.name = name
        self._build_net()
        self.saver = tf.train.Saver()
        self.sess = sess

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 80, 55, 1])
            self.Y = tf.placeholder(tf.float32, [None, 3])

            # Convolution Layer 1 and Pooling Layer 1
            conv1 = tf.layers.conv2d(
                inputs = self.X,
                filters = 32,
                kernel_size = [3, 3],
                padding = 'SAME',
                activation = tf.nn.relu)
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
                inputs = pool1,
                filters = 64,
                kernel_size = [3, 3],
                padding = 'SAME',
                activation = tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(
                inputs = conv2,
                pool_size = [3, 3],
                strides = 2,
                padding = 'SAME')
            dropout2 = tf.layers.dropout(
                inputs = pool2,
                rate = 0.7,
                training = self.training)

            # Convolution Layer 3 and Pooling Layer 3
            conv31 = tf.layers.conv2d(
                inputs = pool2,
                filters = 128,
                kernel_size = [3, 3],
                padding = 'SAME',
                activation = tf.nn.relu)

            # Convolution Layer 3 and Pooling Layer 3
            conv32 = tf.layers.conv2d(
                inputs = conv31,
                filters = 128,
                kernel_size = [3, 3],
                padding = 'SAME',
                activation = tf.nn.relu)

            dropout32 = tf.layers.dropout(
                inputs = conv32,
                rate = 0.7,
                training = self.training)

            # Convolution Layer 3 and Pooling Layer 3
            conv33 = tf.layers.conv2d(
                inputs = conv32,
                filters = 256,
                kernel_size = [3, 3],
                padding = 'SAME',
                activation = tf.nn.relu)
            pool33 = tf.layers.max_pooling2d(
                inputs = conv33,
                pool_size = [2, 2],
                strides = 2,
                padding = 'SAME')
            dropout33 = tf.layers.dropout(
                inputs = pool33,
                rate = 0.7,
                training = self.training)

            # Dense Layer
            flat = tf.reshape(dropout33, [-1, 20 * 14 * 64])
            dense4 = tf.layers.dense(
                inputs = flat,
                units = 512,
                activation = tf.nn.relu)
            dropout4 = tf.layers.dropout(
                inputs = dense4,
                rate = 0.5,
                training = self.training)

            dense41 = tf.layers.dense(
                inputs = dropout4,
                units = 512,
                activation = tf.nn.relu)
            dropout41 = tf.layers.dropout(
                inputs = dense41,
                rate = 0.5,
                training = self.training)

            # Logits Layer (FC 1024 inputs -> 3 outputs)
            self.logits = tf.layers.dense(
                inputs = dropout41,
                units = 3)

    def predict(self, x_test, training = False):
        return self.sess.run(
            self.logits,
            feed_dict = {self.X: x_test, self.training: training})

    def save(self, path):
        self.saver.save(sess=self.sess, save_path=path+self.name)

    def load(self, path):
        self.saver.restore(self.sess, path+self.name)

def get_path(path, num):
    dirt = path+str(num)+'/'
    if not os.path.exists(dirt):
        os.makedirs(dirt)
    return dirt

def get_data(image):
    data = []
    img = cv2.imread(image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (128, 128))
    gray_img = gray_img.reshape(128, 128, 1)
    data.append(gray_img)
    return data

def get_wrinkle(images):
    data = []
    image = cv2.imread(images)
    a = []
    gray_n = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_n = cv2.resize(gray_n, (128, 128))
    # gray_n = gray_n.reshape(128, 128, 1)
    eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('./haarcascade_mcs_nose.xml')

    #######눈 검출#######
    flag = 1
    eyes = eye_cascade.detectMultiScale(gray_n)
    for (ex, ey, ew, eh) in eyes:
        if(flag == 1): #왼쪽 눈
            lex = ex
            ley = ey
            lew = ew
            leh = eh
            flag = flag + 1
        else:       #오른쪽 눈
            rex = ex
            rey = ey
            rew = ew
            reh = eh
        #cv2.rectangle(n, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) #눈 위치 보고싶음 데모ver에서 보셈

    #######눈 교차 오류 해결#######
    if(lex>rex):
        tmp = lex
        lex = rex
        rex = tmp
        tmp = ley
        ley = rey
        rey = tmp
        tmp = lew
        lew = rew
        rew = tmp
        tmp = leh
        leh = reh
        reh = tmp

    #######코 검출#######
    flag = 1
    nose = nose_cascade.detectMultiScale(gray_n)
    for (x, y, w, h) in nose:
        if(flag == 1):
            nx = x
            ny = y
            nw = w
            nh = h
            flag = flag + 1
        #######코 외의 다른 부분이 함께 코로 인식되는 오류 해결#######
        else:
            tmpx = x
            tmpy = y
            tmpw = w
            tmph = h
            if(tmpy>ny):
                nx = tmpx
                ny = tmpy
                nw = tmpw
                nh = tmph
        #cv2.rectangle(img, (nx, ny), (nx+nw, ny+nh), (255, 0, 0), 2) #코 위치 보고싶음 보셈

    #######눈 및 코 꼭지점 지정#######
    LeftEye_point = (int(lex+lew/2) , int(ley+leh/2))
    RightEye_point = (int(rex+rew/2) , int(rey+reh/2))
    Nose_point = (int(nx+nw/2) , int(ny+nh/2))

    #######주름 영역 설정#######
    LeftEye_side_wrinkle = (LeftEye_point[0]-23, LeftEye_point[1]-7, LeftEye_point[0]-13, LeftEye_point[1]+13)      #왼쪽 눈 옆 주름
    RightEye_side_wrinkle = (RightEye_point[0]+13, RightEye_point[1]-7, RightEye_point[0]+23, RightEye_point[1]+13) #오른쪽 눈 옆 주름
    LeftEye_wrinkle = (LeftEye_point[0]-13, LeftEye_point[1]+6, LeftEye_point[0]+13, LeftEye_point[1]+15)           #왼쪽 눈 밑 주름
    RightEye_wrinkle = (RightEye_point[0]-13, RightEye_point[1]+6, RightEye_point[0]+13, RightEye_point[1]+15)      #오른쪽 눈 밑 주름
    LeftNose_wrinkle = (Nose_point[0]-30, Nose_point[1]-18, Nose_point[0]-2, Nose_point[1]+15)                      #왼쪽 팔자주름
    RightNose_wrinkle = (Nose_point[0]+2, Nose_point[1]-18, Nose_point[0]+30, Nose_point[1]+15)                     #오른쪽 팔자주름

    ####### 최후의 예외처리 수단 -> 의도치 않은 곳을 주름영역으로 설정한 경우 혹은 주름영역을 너무 많이 잡아버려 범위를 벗어난 경우 데이터 소각 #######
    if(LeftEye_side_wrinkle[0] < 0 or LeftEye_side_wrinkle[1] < 0 or LeftEye_side_wrinkle[2] > 127 or LeftEye_side_wrinkle[3] > 127 or
       RightEye_side_wrinkle[0] < 0 or RightEye_side_wrinkle[1] < 0 or RightEye_side_wrinkle[2] > 127 or RightEye_side_wrinkle[3] > 127 or
       LeftEye_wrinkle[0] < 0 or LeftEye_wrinkle[1] < 0 or LeftEye_wrinkle[2] > 127 or LeftEye_wrinkle[3] > 127 or
       RightEye_wrinkle[0] < 0 or RightEye_wrinkle[1] < 0 or RightEye_wrinkle[2] > 127 or RightEye_wrinkle[3] > 127 or
       LeftNose_wrinkle[0] < 0 or LeftNose_wrinkle[1] < 0 or LeftNose_wrinkle[2] > 127 or LeftNose_wrinkle[3] > 127 or
       RightNose_wrinkle[0] < 0 or RightNose_wrinkle[1] < 0 or RightNose_wrinkle[2] > 127 or RightNose_wrinkle[3] > 127):
       msg = "사진을 다시 찍어주세요"
       return msg

    #######왼쪽 눈 옆 주름 위에서 11개 까지 -> 공백 58개 -> 오른쪽 눈 옆 주름 위에서 11개 까지#######
    for i in range(LeftEye_side_wrinkle[1], LeftEye_side_wrinkle[1]+10+1):
        for j in range(LeftEye_side_wrinkle[0], LeftEye_side_wrinkle[2]+1):
            a.append(gray_n[i, j])
        for k2 in range(58):
            a.append(0)
        for l in range(RightEye_side_wrinkle[0], RightEye_side_wrinkle[2]+1):
            a.append(gray_n[i, l])

    #######왼쪽 눈 옆 주름 위에서 12번 째 부터 끝까지 -> 왼쪽 눈 밑 주름 -> 공백 4개 -> 오른쪽 눈 밑 주름 -> 오른쪽 눈 옆 주름 위에서 12번 째 부터 끝까지#######
    addNum = 0 #LeftEye_wrinkle 및 RightEye_wrinkle 열 증가를 위한 변수
    for i in range(LeftEye_side_wrinkle[1]+11, LeftEye_side_wrinkle[3]+1):
        for j in range(LeftEye_side_wrinkle[0], LeftEye_side_wrinkle[2]+1):
            a.append(gray_n[i, j])

        for k1 in range(LeftEye_wrinkle[0], LeftEye_wrinkle[2]+1):
            a.append(gray_n[LeftEye_wrinkle[1]+addNum, k1])
        for k2 in range(4):
            a.append(0)
        for k3 in range(RightEye_wrinkle[0], RightEye_wrinkle[2]+1):
            a.append(gray_n[RightEye_wrinkle[1]+addNum, k3])
        addNum+=1

        for l in range(RightEye_side_wrinkle[0], RightEye_side_wrinkle[2]+1):
            a.append(gray_n[i, l])

    #######공백 11개 -> 왼쪽 팔자주름 -> 오른쪽 팔자주름 -> 공백 11개#######
    for i in range(LeftNose_wrinkle[1], LeftNose_wrinkle[3]+1):
        for k2 in range(11):
            a.append(0)
        for j in range(LeftNose_wrinkle[0], LeftNose_wrinkle[2]+1):
            a.append(gray_n[i, j])
        for l in range(RightNose_wrinkle[0], RightNose_wrinkle[2]+1):
            a.append(gray_n[i, l])
        for k2 in range(11):
            a.append(0)
    data1 = np.array(a)
    data1 = data1.reshape(80, 55, 1)
    data.append(data1)
    return data

def predict_sex(path, models):
    data = get_data(path)
    size = len(data)
    predictions = np.zeros(size * 2).reshape(size, 2)
    for m_idx, m in enumerate(models):
        p = m.predict(data)
        predictions += p
    pred = sess.run(tf.argmax(predictions, 1))
    return pred

def predict_age(path, models):
    data = get_wrinkle(path)
    size = len(data)
    predictions = np.zeros(size * 3).reshape(size, 3)
    for m_idx, m in enumerate(models):
        p = m.predict(data)
        predictions += p
    pred = sess.run(tf.argmax(predictions, 1))
    return pred

sess = tf.Session()

sex_models = []
for m in range(num_models):
    sex_models.append(Model(sess, "model" + str(m)))
    save_dir = get_path(sex_path, m)
    sex_models[m].load(save_dir)

# age_models = []
# for m in range(num_models):
#     age_models.append(AModel(sess, "model" + str(m)))
#     save_dir = get_path(age_path, m)
#     age_models[m].load(save_dir)

for line in sys.stdin:
# while True:
    # line = sys.stdin.readline()
    line = line.strip('\n')
    # if line.lower() == 'exit':
    #     print('terminate model')
    #     break
    spred = predict_sex(line, sex_models)
    # apred = predict_age(line, age_models)
    apred = [0]
    if isinstance(apred, str):
        print(apred)
    else:
        if spred[0] == 0:
            smsg = "남성"
        else:
            smsg = "여성"
        if apred[0] == 0:
            amsg = "20~30대"
        elif apred[0] == 1:
            amsg = "40~50대"
        else:
            amsg = "60대 이상"
    print("당신은 %s %s입니다." % (amsg, smsg))
    print()
    sys.stdout.flush()
