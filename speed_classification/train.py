# coding: utf-8

# 数据集：https://serv.cusp.nyu.edu/projects/urbansounddataset/
#
# librosa：https://github.com/librosa/librosa
#
# 分类：
# 0 = air_conditioner
# 1 = car_horn
# 2 = children_playing
# 3 = dog_bark
# 4 = drilling
# 5 = engine_idling
# 6 = gun_shot
# 7 = jackhammer
# 8 = siren
# 9 = street_music

# In[1]:

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import librosa  # pip install librosa
from tqdm import tqdm  # pip install tqdm
import random

# In[2]:

# Parameters
# ==================================================

# Data loading params
# validation数据集占比
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
# 父目录
tf.flags.DEFINE_string("parent_dir", "./", "Data source for the data.")
# 子目录
tf.flags.DEFINE_list("tr_sub_dirs", ['fold1/'], "Data source for the data.")

# Model Hyperparameters
# 第一层输入，MFCC信号
tf.flags.DEFINE_integer("n_inputs", 40, "Number of MFCCs (default: 40)")
# cell个数
tf.flags.DEFINE_integer("n_hidden", 300, "Number of cells (default: 300)")
# 分类数
tf.flags.DEFINE_integer("n_classes", 10, "Number of classes (default: 10)")
# 学习率
tf.flags.DEFINE_float("lr", 0.005, "Learning rate (default: 0.005)")
# dropout参数
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
# 批次大小
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
# 迭代周期
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 100)")
# 多少step测试一次
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
# 多少step保存一次模型
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 500)")
# 最多保存多少个模型
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 2)")

# flags解析
FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()

# 打印所有参数
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# In[3]:

# 获得训练用的wav文件路径列表
def get_wav_files(parent_dir, sub_dirs):
    wav_files = []
    for sub_dir in sub_dirs:
        for dirpath, dirnames, filenames in os.walk(parent_dir + sub_dir):
            for filename in filenames:
                if filename.endswith('.wav') or filename.endswith('.WAV'):
                    filename_path = os.sep.join([dirpath, filename])
                    wav_files.append(filename_path)
    return wav_files


# def get_wav_files(parent_dir,sub_dirs):
#     wav_files = []
#     for l, sub_dir in enumerate(sub_dirs):
#
#         wav_path = os.path.join(parent_dir, sub_dir)
#         for (dirpath, dirnames, filenames) in os.walk(wav_path):
#             for filename in filenames:
#                 if filename.endswith('.wav') or filename.endswith('.WAV'):
#                     filename_path = os.sep.join([dirpath, filename])
#                     wav_files.append(filename_path)
#     return wav_files

# 获取文件mfcc特征和对应标签
def extract_features(wav_files):
    inputs = []
    labels = []

    for wav_file in tqdm(wav_files):
        # 读入音频文件
        audio, fs = librosa.load(wav_file)

        # 获取音频mfcc特征
        # [n_steps, n_inputs]
        mfccs = np.transpose(librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=FLAGS.n_inputs), [1, 0])
        inputs.append(mfccs.tolist())
        # 获取label
    for wav_file in wav_files:
        label = wav_file.split('/')[-1].split('-')[1]
        labels.append(label)
    return inputs, np.array(labels, dtype=np.int)


# In[4]:

# 获得训练用的wav文件路径列表
wav_files = get_wav_files(FLAGS.parent_dir, FLAGS.tr_sub_dirs)
# # 获取文件mfcc特征和对应标签
tr_features, tr_labels = extract_features(wav_files)

np.save('tr_features.npy', tr_features)
np.save('tr_labels.npy', tr_labels)

# tr_features=np.load('tr_features.npy',allow_pickle=True)
# tr_labels=np.load('tr_labels.npy',allow_pickle=True)

wavFile = "./fold5/7061-6-0-0.wav"
test = []
test.append(wavFile)
test_features, test_labels = extract_features(test)
# In[5]:

# (batch,step,input)
# (50,173,40)

# 计算最长的step
wav_max_len = max([len(feature) for feature in tr_features])
print("max_len:", wav_max_len)


# 填充0
def padding(trest_features):
    xxx_data = []
    for mfccs in trest_features:
        while len(mfccs) < wav_max_len:  # 只要小于wav_max_len就补n_inputs个0
            mfccs.append([0] * FLAGS.n_inputs)
        xxx_data.append(mfccs)
    return xxx_data


tr_data = []
tr_data = padding(tr_features)
tr_data = np.array(tr_data)

test_data = padding(test_features)
test_data = np.array(test_data)
# In[6]:

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(tr_data)))
x_shuffled = tr_data[shuffle_indices]
y_shuffled = tr_labels[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
# 数据集切分为两部分
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_shuffled)))
train_x, test_x = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
train_y, test_y = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]


# In[7]:
def train_model():
    # placeholder
    x = tf.placeholder("float", [None, wav_max_len, FLAGS.n_inputs], name="x")
    y = tf.placeholder("float", [None], name="y")
    dropout = tf.placeholder(tf.float32)
    # learning rate
    lr = tf.Variable(FLAGS.lr, dtype=tf.float32, trainable=False)

    # 定义RNN网络
    # 初始化权制和偏置
    weights = tf.Variable(tf.truncated_normal([FLAGS.n_hidden, FLAGS.n_classes], stddev=0.1), name="weights")
    biases = tf.Variable(tf.constant(0.1, shape=[FLAGS.n_classes]), name="biases")

    # 多层网络
    num_layers = 3

    def grucell():
        cell = tf.contrib.rnn.GRUCell(FLAGS.n_hidden)
        #     cell = tf.contrib.rnn.LSTMCell(FLAGS.n_hidden)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        return cell

    cell = tf.contrib.rnn.MultiRNNCell([grucell() for _ in range(num_layers)])

    outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    # 预测值
    prediction = tf.nn.softmax(tf.matmul(final_state[0], weights) + biases, name="prediction")
    tf.add_to_collection("prediction", prediction)

    # labels转one_hot格式
    one_hot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=FLAGS.n_classes)

    # loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=one_hot_labels))

    # optimizer
    with tf.name_scope('train_op'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(one_hot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Initializing the variables
    init = tf.global_variables_initializer()
    # 定义saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("./log", sess.graph)
        # Generate batches
        batches = batch_iter(list(zip(train_x, train_y)), FLAGS.batch_size, FLAGS.num_epochs)

        for i, batch in enumerate(batches):
            i = i + 1
            x_batch, y_batch = zip(*batch)
            sess.run([optimizer], feed_dict={x: x_batch, y: y_batch, dropout: FLAGS.dropout_keep_prob})

            # 测试
            if i % FLAGS.evaluate_every == 0:
                sess.run(tf.assign(lr, FLAGS.lr * (0.99 ** (i // FLAGS.evaluate_every))))
                learning_rate = sess.run(lr)
                tr_acc, _loss = sess.run([accuracy, cross_entropy], feed_dict={x: train_x, y: train_y, dropout: 1.0})
                ts_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y, dropout: 1.0})
                print("Iter {}, loss {:.5f}, tr_acc {:.5f}, ts_acc {:.5f}, lr {:.5f}".format(i, _loss, tr_acc, ts_acc,
                                                                                             learning_rate))

                # 保存模型

                # if i % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, "./model", global_step=i)
                print("Saved model checkpoint to {}\n".format(path))

        tf.saved_model.simple_save(sess,
                                   "speed-model",
                                   inputs={"Input": x},
                                   outputs={"Output": prediction})


# In[8]:

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
        Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    # 每个epoch的num_batch
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("num_batches_per_epoch:", num_batches_per_epoch)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# In[9]:

def load_model():
    # 测试数据构造：模拟2张32x32的RGB图
    X = np.array(np.arange(6144, 12288)).reshape(2, 32, 32, 3)  # 2:张，32*32：图片大小，3：RGB
    Y = [3, 1]
    Y = np.array(Y)
    X = X.astype('float32')
    X = np.multiply(X, 1.0 / 255.0)

    with tf.Session() as sess:
        # 加载元图和权重
        saver = tf.train.import_meta_graph('./model-100.meta')
        saver.restore(sess, tf.train.latest_checkpoint("./"))
        print("load success")
        graph = tf.get_default_graph()
        pred_y = tf.get_collection("prediction")

        feed_dict = {"x:0": test_data}
        pred = sess.run(pred_y, feed_dict)[0]
        print('pred:', pred, '\n')  # pred是新数据下得到的预测值
        pred = sess.run(tf.argmax(pred, 1))
        print("the predict is: ", pred)


# In[ ]:
# train_model()
load_model()


