import scipy.io as sio
import numpy as np

import tensorflow as tf
import time

from feature_extraction import feature_extraction
from network import network
from sum_rate import np_sum_rate

N = 10
num_train = 100000
num_test = 10000
epochs = 1
batch_size = 256

load = sio.loadmat('data/Train_data_%d_%d.mat' % (N, num_train))
loadTest = sio.loadmat('data/Test_data_%d_%d.mat' % (N, num_test))
Htrain = load['Xtrain']
Ptrain = load['Ytrain']
H_test = loadTest['X']
P_test = loadTest['Y']
timeW = loadTest['swmmsetime']
swmmsetime = timeW[0, 0]

H_ii, H_ij, H_ij_T, D = feature_extraction(Htrain,num_train,N)

weights = {
    'w_c_1': tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.1)),
    'w_c_2': tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.1)),
    'w_c_3': tf.Variable(tf.random_normal([3, 3, 32, 16], stddev=0.1)),
    'w_c_4': tf.Variable(tf.random_normal([3, 3, 16, 6], stddev=0.1)),

    'w_fc_1': tf.Variable(tf.random_normal([12, 40], stddev=0.1)),
    'w_fc_2': tf.Variable(tf.random_normal([40, 20], stddev=0.1)),
    'w_fc_3': tf.Variable(tf.random_normal([20, 1])),
}

biases = {
    'b_c_1': tf.Variable(tf.random_normal([32], stddev=0.1)),
    'b_c_2': tf.Variable(tf.random_normal([32], stddev=0.1)),
    'b_c_3': tf.Variable(tf.random_normal([16], stddev=0.1)),
    'b_c_4': tf.Variable(tf.random_normal([6], stddev=0.1)),

    'b_fc_1': tf.Variable(tf.random_normal([40], stddev=0.1)),
    'b_fc_2': tf.Variable(tf.random_normal([20], stddev=0.1)),
    'b_fc_3': tf.Variable(tf.random_normal([1])),

}

X_ij = tf.placeholder(tf.float32, [None, N, N, 1]) #Xintert
X_ij_T = tf.placeholder(tf.float32, [None, N, N, 1]) #Xinterf
X_D = tf.placeholder(tf.float32, [None, N, N, 1]) #Xdiag_o

y = tf.placeholder(tf.float32, [None, N, 1])

inp = tf.concat((X_ij_T, X_ij, X_D), axis=3)

pred = network(inp, weights, biases)

cost2 = tf.reduce_mean(tf.square(pred - y)) #Mean Square Error
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost2)

valid_split = 0.1

total_sample_size = num_train
validation_sample_size = int(total_sample_size*valid_split)
training_sample_size = total_sample_size - validation_sample_size


H_ii_train = np.reshape(H_ii[ 0:training_sample_size, :], (training_sample_size, N, 1))
H_ij_train = np.reshape(H_ij[ 0:training_sample_size, :, :], (training_sample_size, N, N, 1))
H_ij_T_train = np.reshape(H_ij_T[ 0:training_sample_size, :, :], (training_sample_size, N, N, 1))
D_train = np.reshape(D[ 0:training_sample_size, :, :], (training_sample_size, N, N, 1))

P_train = np.reshape(Ptrain[0:training_sample_size, :],  (training_sample_size, N, 1))

H_ii_val = np.reshape(H_ii[ training_sample_size:total_sample_size, :], (validation_sample_size, N, 1))
H_ij_val = np.reshape(H_ij[ training_sample_size:total_sample_size, :, :], (validation_sample_size, N, N, 1))
H_ij_T_val = np.reshape(H_ij_T[ training_sample_size:total_sample_size, :, :], (validation_sample_size, N, N, 1))
D_val = np.reshape(D[ training_sample_size:total_sample_size, :, :], (validation_sample_size, N, N, 1))

P_val = np.reshape(Ptrain[training_sample_size:total_sample_size, :],  (validation_sample_size, N, 1))


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
start_time = time.time()

MSETime=np.zeros((epochs, 3))

total_batch = int(total_sample_size / batch_size)

for epoch in range(epochs):
    for ii in range(total_batch):
        batch = np.random.randint(training_sample_size, size=batch_size)
        _, cost_value = sess.run([optimizer, cost2],
                                 feed_dict={X_ij: H_ij_train[batch, :, :], X_ij_T: H_ij_T_train[batch, :, :], X_D: D_train[batch, :, :], y: P_train[batch, :] })
        if (ii % 50 == 0):
            print("#", end='')
            # print('')

    cost_value2 = sess.run(cost2,
                           feed_dict={X_ij: H_ij_val, X_ij_T: H_ij_T_val, X_D: D_val, y: P_val})
    MSETime[epoch, 0] = cost_value
    MSETime[epoch, 1] = cost_value2
    MSETime[epoch, 2] = epoch + 1

    print("\n", epoch + 1, cost_value, cost_value2, time.time() - start_time)

sio.savemat('CNN_MSE_%d_%d.mat'%(N, num_train), {'MSETime': MSETime})



H_ii_t, H_ij_t, H_ij_T_t, D_t = feature_extraction(H_test, num_test, N)
H_ii_t = np.reshape(H_ii_t, (num_test, N, 1))
H_ij_t = np.reshape(H_ij_t, (num_test, N, N, 1))
H_ij_T_t = np.reshape(H_ij_T_t, (num_test, N, N, 1))
D_t = np.reshape(D_t, (num_test, N, N, 1))
P_t = np.reshape(P_test,  (num_test, N, 1))

start_time = time.time()
cost_value = sess.run(pred, feed_dict={X_ij: H_ij_t, X_ij_T: H_ij_T_t, X_D: D_t, y: P_t})
pred_time = time.time()-start_time
pred_y = np.reshape(cost_value,(num_test,N))

sum_rate_cnn = np_sum_rate(H_test,pred_y)*np.log2(np.exp(1))
sum_rate_swmmse = np_sum_rate(H_test,P_test)*np.log2(np.exp(1))

print('sum rate for CNN', sum_rate_cnn)
print('sum rate for SWMMSE', sum_rate_swmmse)
print("%f%% in %f sec time" % (sum_rate_cnn/sum_rate_swmmse*100, pred_time))
print(swmmsetime)