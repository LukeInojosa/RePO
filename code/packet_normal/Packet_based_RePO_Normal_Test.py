import tensorflow as tf
import numpy as np
import os
import sys
import time
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model


def get_files(day,prefix = '../data/packet_based/'):
    all_files = []
    prefix = prefix+day
    for file in os.listdir(prefix):
        if file.endswith(".npy") and file.startswith('part'):
            all_files.append(os.path.join(prefix, file))
    all_files = sorted(all_files)
    return all_files

def get_labels(timesteps=20):
    # We move the window 20 steps forward at each time.
    Y_test = []
    for day in ['tuesday','wednesday','thursday','friday']:
        temp = np.load('../data/packet_based/'+day+'/labels.npy')
        Y_test.append(temp)
    Ys = []
    for yt in Y_test:
        a = timesteps -  len(yt) % timesteps
        temp = yt[:a]
        y_test = np.concatenate((yt,temp),axis=0)
        y_test = y_test.reshape(-1,timesteps)
        y_test = y_test[:,-1]
        Ys.append(y_test)
    all_labels = np.concatenate(Ys,axis=0)
    return all_labels

@tf.function
def test_step(x):
    mask = tf.random.uniform(shape=[50*100,timesteps,num_input])
    mask = tf.cast((mask>0.75),tf.float32)
    partial_x = mask*x
    rec_x = model(partial_x, training=False)
    score = tf.reduce_mean(tf.square(rec_x - x),axis=[1,2])
    return score


def load_and_predict_with_repo(day):
    all_files = get_files(day)
    x_test = []
    for f in all_files:
        print (f)
        x_test.append(np.load(f))
    x_test = np.concatenate(x_test,axis=0)
    x_test = (x_test - train_min)/(train_max - train_min+0.000001)
    a = timesteps -  len(x_test) % timesteps
    temp = x_test[:a]
    x_test = np.concatenate((x_test,temp),axis=0)
    x_test = x_test.reshape(-1,timesteps*num_input)
    x_test = x_test.astype(np.float32)
    score_np = np.zeros(len(x_test))
    begin_time = time.time()
    batch_size = 50*100
    for i in range(0,len(x_test),batch_size):
        if i%100000==0:
            print(i,time.time() - begin_time)
        sample = x_test[i:i+batch_size]
        if len(sample)<batch_size:
            break
        sample = sample.reshape(-1,timesteps,num_input)
        rec_error = test_step(sample)
        try:
            score_np[i:i+batch_size] = rec_error.numpy()
        except:
            pass
    total_time = time.time() - begin_time
    print (i,total_time)
    return score_np


def load_and_predict_with_repo_plus(day):
    all_files = get_files(day)
    x_test = []
    for f in all_files:
        print (f)
        x_test.append(np.load(f))
    x_test = np.concatenate(x_test,axis=0)
    x_test = (x_test - train_min)/(train_max - train_min+0.000001)
    a = timesteps -  len(x_test) % timesteps
    temp = x_test[:a]
    x_test = np.concatenate((x_test,temp),axis=0)
    x_test = x_test.reshape(-1,timesteps*num_input)
    x_test = x_test.astype(np.float32)
    score_np = np.zeros(len(x_test))
    begin_time = time.time()
    plus_batch_size = 50
    for i in range(0,len(x_test),plus_batch_size):
        if i%100000==0:
            print(i,time.time() - begin_time)
        sample = x_test[i:i+plus_batch_size]
        if len(sample)<plus_batch_size:
            break
        sample = sample.reshape(-1,1,timesteps,num_input) +  np.zeros((100,timesteps,num_input),np.float32)
        sample = sample.reshape(-1,timesteps,num_input)
        rec_error = test_step(sample)
        rec_error_np = rec_error.numpy().reshape(-1,5,20)
        best_rec_err_val =  np.min(rec_error_np,axis=-1)
        try:
            score_np[i:i+plus_batch_size] = np.sum(best_rec_err_val,axis=1)
        except:
            pass
    total_time = time.time() - begin_time
    print (i,total_time)
    return score_np

label_names = ['Benign','FTP-Patator','SSH-Patator','Slowloris','Slowhttptest','Hulk','GoldenEye','Heartbleed', 'Web-Attack', 'Infiltration','Botnet','PortScan','DDoS']

train_min = np.load('../data/packet_based/x_train_meta/train_min.npy')
train_max = np.load('../data/packet_based/x_train_meta/train_max.npy')

timesteps = 20
num_input = 29

model = tf.keras.models.load_model('../models/pkt_model/')

all_labels = get_labels()

predicted_scores = []
for day in ['tuesday','wednesday','thursday','friday']:
    score_temp = load_and_predict_with_repo_plus(day)
    predicted_scores.append(score_temp)

predicted_scores = np.concatenate(predicted_scores,axis=0)
real_labels = all_labels!=-1
all_scores = predicted_scores[real_labels]
all_labels = all_labels[real_labels]
print (all_labels.shape,all_scores.shape)

fpr = 0.01
benign_scores_sorted = np.sort(all_scores[all_labels==0])
thr_ind = benign_scores_sorted.shape[0]*fpr
thr_ind = int(np.round(thr_ind))
thr = benign_scores_sorted[-thr_ind]
print (thr)

for i in range(len(label_names)):
    #### Exclude web attacks from results
    if label_names[i]=='Web-Attack':
        continue
    scores = all_scores[all_labels==i]
    if i==0:
        fpr = "{0:0.4f}".format(np.sum(scores>=thr)/(0. + len(scores)))
        print('FPR:',fpr)
    else:
        tpr = "{0:0.4f}".format(np.sum(scores>=thr)/(0. + len(scores)))
        print(label_names[i]+':',tpr)

#adicionado para calcular o threshold que dá um fpr de 0.1 para o teste adversarial
fpr = 0.1
thr_ind = benign_scores_sorted.shape[0]*fpr
thr_ind = int(np.round(thr_ind))
thr = benign_scores_sorted[-thr_ind]
print("Treshold para 0.1 fpr:")
print (thr)
