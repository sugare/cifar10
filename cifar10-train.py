import tensorflow as tf
import  os
import numpy as np
import pickle

# 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 参数
tf.app.flags.DEFINE_string('data_dir', 'cifar-10-batches-py','dataset dir')
tf.app.flags.DEFINE_string('model_dir', 'models', 'model save dir')
#tf.app.flags.DEFINE_string('log_dir', 'log', 'events save dir')

FLAGS = tf.app.flags.FLAGS

# 文件存放目录
CIFAR_DIR = FLAGS.data_dir
CIFAR_MODEL_DIR = FLAGS.model_dir
CIFAR_LOG_DIR = "./log" #FLAGS.log_dir

print("CIFAR_DIR: ", CIFAR_DIR)
print("CIFAR_MODEL_DIR: ", CIFAR_MODEL_DIR)
print("CIFAR_LOG_DIR", CIFAR_LOG_DIR)

def load_data( filename ):
    '''read data from data file'''
    with open( filename, 'rb' ) as f:
        data = pickle.load( f, encoding='bytes' ) # python3 需要添加上encoding='bytes'
        return data[b'data'], data[b'labels'] # 并且 在 key 前需要加上 b

class CifarData:
    def __init__( self, filenames, need_shuffle ):
        '''参数1:文件夹 参数2:是否需要随机打乱'''
        all_data = []
        all_labels = []

        for filename in filenames:
            # 将所有的数据,标签分别存放在两个list中
            data, labels = load_data( filename )
            all_data.append( data )
            all_labels.append( labels )

        # 将列表 组成 一个numpy类型的矩阵!!!!
        self._data = np.vstack(all_data)
        # 对数据进行归一化, 尺度固定在 [-1, 1] 之间
        self._data = self._data / 127.5 - 1
        # 将列表,变成一个 numpy 数组
        self._labels = np.hstack( all_labels )
        # 记录当前的样本 数量
        self._num_examples = self._data.shape[0]
        # 保存是否需要随机打乱
        self._need_shuffle = need_shuffle
        # 样本的起始点
        self._indicator = 0
        # 判断是否需要打乱
        if self._need_shuffle:
            self._shffle_data()

    def _shffle_data( self ):
        # np.random.permutation() 从 0 到 参数,随机打乱
        p = np.random.permutation( self._num_examples )
        # 保存 已经打乱 顺序的数据
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch( self, batch_size ):
        '''
        获取数据
        :param batch_size:  数据量
        :return:  样本 标签
        '''
        # 开始点 + 数量 = 结束点
        end_indictor = self._indicator + batch_size
        # 如果结束点大于样本数量
        if end_indictor > self._num_examples:
            if self._need_shuffle:
                # 重新打乱
                self._shffle_data()
                # 开始点归零,从头再来
                self._indicator = 0
                # 重新指定 结束点. 和上面的那一句,说白了就是重新开始
                end_indictor = batch_size # 其实就是 0 + batch_size, 把 0 省略了
            else:
                raise Exception( "have no more examples" )
        # 再次查看是否 超出边界了
        if end_indictor > self._num_examples:
            raise Exception( "batch size is larger than all example" )

        # 把 batch 区间 的data和label保存,并最后return
        batch_data = self._data[self._indicator:end_indictor]
        batch_labels = self._labels[self._indicator:end_indictor]
        self._indicator = end_indictor
        return batch_data, batch_labels
def main(_):
# 拿到所有文件名称
    train_filename = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
    # 拿到标签
    test_filename = [os.path.join(CIFAR_DIR, 'test_batch')]
    
    # 拿到训练数据和测试数据
    train_data = CifarData( train_filename, True )
    test_data = CifarData( test_filename, False )
    
    # 设计计算图
    # 形状 [None, 3072] 3072 是 样本的维数, None 代表位置的样本数量
    x = tf.placeholder( tf.float32, [None, 3072] )
    # 形状 [None] y的数量和x的样本数是对应的
    y = tf.placeholder( tf.int64, [None] )
    
    # 因为现在是 10分类,需要10个神经元,权重是10列
    w = tf.get_variable( 'w', [x.get_shape()[-1], 10], initializer= tf.random_normal_initializer(0, 1))
    # 同样,神经元也需要10个
    b = tf.get_variable( 'b', [10], initializer=tf.constant_initializer(0.0))
    
    # 修改 (None, 1) 计算wx + b (None, 3072) * (3072, 10) = (None, 10)
    y_ = tf.matmul( x, w ) + b
    
    '''
    # 将输出结果转化为概率
    p_y = tf.nn.softmax( y_ )
    # 因为原始数据都是用0-9来表示,需要转化成one_hot码来进行计算
    y_one_hot = tf.one_hot( y, 10, dtype=tf.float32 )
    # 设定损失函数
    loss = tf.reduce_mean( tf.square( y_one_hot - p_y ) )
    '''
    # 使用交叉熵 设置损失函数
    loss = tf.losses.sparse_softmax_cross_entropy(labels = y, logits = y_)
    # 该api,做了三件事儿 1. y_ -> softmax 2. y -> one_hot 3. loss = ylogy
    
    # 预测值 获得的是 每一行上 最大值的 索引.注意:tf.argmax()的用法,其实和 np.argmax() 一样的
    predict = tf.argmax( y_, 1 )
    # 将布尔值转化为int类型,也就是 0 或者 1, 然后再和真实值进行比较. tf.equal() 返回值是布尔类型
    correct_prediction = tf.equal( predict, y )
    # 比如说第一行最大值索引是6,说明是第六个分类.而y正好也是6,说明预测正确
    
    
    
    # 将上句的布尔类型 转化为 浮点类型,然后进行求平均值,实际上就是求出了准确率
    accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float64) )
    
    with tf.name_scope( 'train_op' ): # tf.name_scope() 定义命名空间
        train_op = tf.train.AdamOptimizer( 1e-3 ).minimize( loss ) # 将 损失函数 降到 最低
    
    # 初始化变量
    init = tf.global_variables_initializer()
    
    batch_size = 20
    train_steps = 1000
    test_steps = 100
    with tf.Session() as sess:
        sess.run( init ) # 注意: 这一步必须要有!!
        writer = tf.summary.FileWriter(CIFAR_LOG_DIR,sess.graph)
        # 开始训练
        for i in range( train_steps ):
            # 得到batch
            batch_data, batch_labels = train_data.next_batch( batch_size )
            # 获得 损失值, 准确率
            loss_val, acc_val, _ = sess.run( [loss, accuracy, train_op], feed_dict={x:batch_data, y:batch_labels} )
            # 每 500 次 输出一条信息
            if ( i+1 ) % 500 == 0:
                print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % ( i+1, loss_val, acc_val ))
            # 每 5000 次 进行一次 测试
            if ( i+1 ) % 5000 == 0:
                # 获取数据集,但不随机
                test_data = CifarData( test_filename, False )
                all_test_acc_val = []
                for j in range( test_steps ):
                    test_batch_data, test_batch_labels = test_data.next_batch( batch_size )
                    test_acc_val = sess.run( [accuracy], feed_dict={ x:test_batch_data, y:test_batch_labels } )
                    all_test_acc_val.append( test_acc_val )
                test_acc = np.mean( all_test_acc_val )
                print('[Test ] Step: %d, acc: %4.5f' % ( (i+1), test_acc ))
        tf.saved_model.simple_save(sess,
                               CIFAR_MODEL_DIR,
                               inputs={"Input": x},
                               outputs={"Output": y})
if __name__=="__main__":
    tf.app.run()
