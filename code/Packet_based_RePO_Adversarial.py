import tensorflow as tf
import numpy as np
import os
import sys
import time
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

### At this threshold the FPR of the model we trained is 0.1
thr = 0.07062649726867676

attack_to_label={
    'FTP-Patator':1,'SSH-Patator':2, #Tuesday attacks
    'Slowloris':3,'Slowhttptest':4,'Hulk':5,'GoldenEye':6,'Heartbleed':7, #Wednesday attacks
    'Web-Attack':8, 'Infiltration':9, #Thursday attacks
    'Botnet':10,'PortScan':11,'DDoS':12 #Friday attacks
}

def get_files(day,prefix = '../data/packet_based/'):
    all_files = []
    prefix = prefix+day
    for file in os.listdir(prefix):
        if file.endswith(".npy") and file.startswith('part'):
            all_files.append(os.path.join(prefix, file))
    all_files = sorted(all_files)
    return all_files

def get_test_set(day):
    train_min = np.load('../data/packet_based/x_train_meta/train_min.npy')
    train_max = np.load('../data/packet_based/x_train_meta/train_max.npy')

    all_files = get_files(day)
    x_test = []
    for f in all_files:
        print (f)
        x_test.append(np.load(f))
    x_test = np.concatenate(x_test,axis=0)

    yt = np.load('../data/packet_based/'+day+'/labels.npy')
    y_test = yt
    return x_test,y_test,train_min,train_max


@tf.function
def test_step(x):
    def_mask = tf.random.uniform(shape=[1*100,timesteps,num_input])
    def_mask = tf.cast((def_mask>0.75),tf.float32)
    x_normalized =(x - train_min)/(train_max - train_min+0.000001)

    partial_x = def_mask*x_normalized
    rec_x = model(partial_x, training=False)
    score = tf.reduce_mean(tf.square(rec_x - x_normalized),axis=[1,2])
    score = tf.reduce_min(tf.reshape(score,[5,20]),axis=-1)
    score = tf.reduce_sum(score)
    return score




# Crafiting adversarial Examples:

@tf.function
def get_delayed_splited(x,p_len):
    
    alpha_split_2 = tf.zeros((1,1,num_input)) + alpha_split
    alpha_split_2 = tf.minimum(alpha_split_2,0.)
    alpha_split_2 = tf.maximum(alpha_split_2,-p_len+np.float32(61))
    alpha_delay_2 = tf.maximum(alpha_delay,0.)
    alpha_delay_2 = tf.minimum(alpha_delay_2,15.)
    mask = np.ones((1,1,29))
    masked_alpha = alpha_delay_2*mask
    mask_split = np.zeros((1,1,29))
    mask_split[0,0,1] = mask_split[0,0,3] = 1
    mask_split = mask_split.astype(np.bool)
    alpha_final = tf.where(mask_split,alpha_split_2,masked_alpha)

    last_ts_modified = x[0,-1]+alpha_final
    adv_x = tf.concat((x[:,:19],last_ts_modified),axis=1)
    return adv_x

@tf.function
def delay_split_optim(x,p_len):
    

    with tf.GradientTape() as tape:
        adv_x = get_delayed_splited(x,p_len)
        adv_x_normalized = (adv_x- train_min)/(train_max - train_min+0.000001)
        rand_mask = tf.random.uniform(shape=[100,timesteps,num_input])
        rand_mask = tf.cast((rand_mask>0.75),tf.float32)
        partial_adv_x_n = adv_x_normalized*rand_mask
        rec_adv_x_n = model(partial_adv_x_n,training=False)
        score1_split = tf.reduce_mean(tf.square(rec_adv_x_n - adv_x_normalized),axis=[1,2])
        score1_split = tf.reduce_sum(score1_split)
        loss_split = score1_split

    gradients = tape.gradient(loss_split, [alpha_delay,alpha_split])
    optimizer.apply_gradients(zip(gradients, [alpha_delay,alpha_split]))


@tf.function
def get_injected(x,inject_mask):

    packet_mins = [0,60,20,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1]
    packet_maxs = [15,2**16,20,2**16,1,1,1,255,2**16,2**16,2**32,2**32,1,1,1,1,1,1,1,1,1,1,2**16,2**16,2**16,2**16,2**16,0,1]
    alpha_inject_2 = tf.minimum(alpha_inject,packet_maxs)
    alpha_inject_2 = tf.maximum(alpha_inject_2,packet_mins)
    alpha_inject_masked= alpha_inject_2*inject_mask
    adv_x1 = tf.concat((x[:,:19],alpha_inject_masked),axis=1)
    adv_x2 = tf.concat((x[:,1:19],alpha_inject_masked,x[:,19:]),axis=1)
    return adv_x1,adv_x2


@tf.function
def inject_optim(x,inject_mask):
    
    with tf.GradientTape() as tape:
        adv_x1,adv_x2 = get_injected(x,inject_mask)
        adv_x1_normalized = (adv_x1- train_min)/(train_max - train_min+0.000001)
        adv_x2_normalized = (adv_x2- train_min)/(train_max - train_min+0.000001)
        rand_mask = tf.random.uniform(shape=[100,timesteps,num_input])
        rand_mask = tf.cast((rand_mask>0.75),tf.float32)
        partial_adv_x_n1 = adv_x1_normalized*rand_mask
        rand_mask = tf.random.uniform(shape=[100,timesteps,num_input])
        rand_mask = tf.cast((rand_mask>0.75),tf.float32)
        partial_adv_x_n2 = adv_x2_normalized*rand_mask
        rec_adv_x_n1 = model(partial_adv_x_n1,training=False)
        rec_adv_x_n2 = model(partial_adv_x_n2,training=False)
        score1_inject1 = tf.reduce_mean(tf.square(rec_adv_x_n1 - adv_x1_normalized),axis=[1,2])
        score1_inject1 = tf.reduce_sum(score1_inject1)
        score1_inject2 = tf.reduce_mean(tf.square(rec_adv_x_n2 - adv_x2_normalized),axis=[1,2])
        score1_inject2 = tf.reduce_sum(score1_inject2)
        loss_inject = score1_inject1 + score1_inject2

    gradients = tape.gradient(loss_inject, [alpha_inject])
    optimizer.apply_gradients(zip(gradients, [alpha_inject]))


def find_adv(x):

    sc = test_step(x)
    sc = sc.numpy()
    if sc<thr:
        return 'cons_as_ben'
    if x[0,-1,-1]==2: #packet is sent from victim
        return inject(np.copy(x))
    
    #packet is sent from attacker
    res = delay_and_split(np.copy(x))
    if res and len(res)>0:
        return ('split',res)
    i_res = inject(np.copy(x))
    return i_res


def delay_and_split(x):

    alpha_delay.assign(np.zeros(alpha_delay.shape))
    alpha_split.assign(np.zeros(alpha_split.shape))
    len_last = x[0,-1,1]
    ip_len_last = x[0,-1,3]
    adv_x = get_delayed_splited(x,len_last)
    adv_x = adv_x.numpy()
    sc = test_step(adv_x)
    sc = sc.numpy()
    if sc<thr:
        return [adv_x]
    res = []
    for i in range(300):
        delay_split_optim(x,len_last)
        adv_x = get_delayed_splited(x,len_last)
        adv_x = adv_x.numpy()
        adv_x[0,-1,1:4] = np.round(adv_x[0,-1,1:4])
        sc = test_step(adv_x)
        sc = sc.numpy()
        if sc<thr:
            first_part = np.copy(adv_x)
            diff = len_last - adv_x[0,-1,1]
            if diff>0:
                adv_x[0,-1,1] = diff + 60
                adv_x[0,-1,0] = 0
                adv_x[0,-1,3] = diff + 60 - 14 #14 is the frame header len.
                second_part = delay_and_split(adv_x)
                if second_part==None:
                    return None
                res.append(first_part)
                res.extend(second_part)
            else:
                res.append(first_part)
            break
    if len(res)==0:
        return None
    return res


tcp_mask = [1]*8 + [1]*16 + [0]*3 + [0]*1 + [1]*1
udp_mask = [1]*8 + [0]*16 + [1]*3 + [0]*1 + [1]*16
def inject(x,mask_type = 'tcp'):
    alpha_inject.assign(np.zeros(alpha_inject.shape))
    cur_mask = tcp_mask if mask_type=='tcp' else udp_mask
    for i in range(300):
        inject_optim(x,cur_mask)
        adv_x1,adv_x2 = get_injected(x,cur_mask)
        adv_x1,adv_x2  = adv_x1.numpy(),adv_x2.numpy()
        adv_x1[0,:,1:] = np.round(adv_x1[0,:,1:])
        sc = test_step(adv_x1)
        sc = sc.numpy()
        adv_x2[0,:,1:] = np.round(adv_x2[0,:,1:])
        sc2 = test_step(adv_x2)
        sc2 = sc2.numpy()
        if sc<thr and sc2<thr: #fooled
            fake_packets.append(adv_x1[0,-1])
            return ('inject',adv_x1[0,-1]) #<--- the packet which is inject should be returned
    res = None
    return res



timesteps = 20
model = tf.keras.models.load_model('../models/pkt_model/')
attack_types = ['Slowloris','Slowhttptest','Hulk','GoldenEye','Heartbleed', 'Web-Attack', 'Infiltration','Botnet','PortScan','DDoS']

for attack_type in attack_types:

    if attack_type in ['FTP-Patator','SSH-Patator']:
        day = 'tuesday'
    elif attack_type in ['Slowloris','Slowhttptest','Hulk','GoldenEye','Heartbleed']:
        day = 'wednesday'
    elif attack_type in ['Web-Attack', 'Infiltration']:
        day = 'thursday'
    else:
        day = 'friday'

    x_test,y_test,train_min,train_max = get_test_set(day)
    num_input = x_test.shape[1]
    print(x_test.shape,y_test.shape)


    alpha_delay = tf.Variable(np.zeros((1, 1,num_input),dtype=np.float32),name='delay')
    alpha_split = tf.Variable(np.zeros((1),dtype=np.float32),name='split')
    alpha_inject = tf.Variable(np.zeros((1, 1,num_input), dtype=np.float32),name='modifier')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    attack_label = attack_to_label[attack_type]
    x_test_mal = x_test[y_test==attack_label]
    x_test_mal = np.concatenate((np.zeros((timesteps-1,num_input)),x_test_mal),axis=0)
    print (x_test_mal.shape)
    x_test_mal = x_test_mal[:2500].astype(np.float32)
    score_np = np.zeros(len(x_test_mal))
    st = timesteps-1
    begin_time = time.time()
    for i in range(len(x_test_mal)-timesteps):
        if i%10000==0:
            print (i,time.time() - begin_time)
        sample = x_test_mal[i:i+timesteps][None]
        score_temp = test_step(sample)
        score_np[st+i] = score_temp.numpy()
    print (i,time.time() - begin_time)

    mal_scores = score_np[timesteps:]
    print ("TPR in normal setting for "+attack_type+" is {0:0.4f}".format(np.sum(mal_scores>=thr)/len(mal_scores)))


    stream = []
    stream_status = []
    fake_packets = []
    cons_as_mal = 0
    cons_as_ben = 0
    fooled = 0
    x = []
    begin_time = time.time()
    for i in range(timesteps-1):
        stream.append(x_test_mal[i])
        stream_status.append(None)
    for i in range(timesteps-1,len(x_test_mal)):
        if i%100==0:
            print ('#',i,(time.time() - begin_time)/60.,cons_as_mal,cons_as_ben,fooled)
        x = np.zeros((1,20,29),dtype=np.float32)
        x[0,:19] = np.array(stream[-19:])
        x[0,19] = x_test_mal[i]
        temp = find_adv(np.copy(x))
        stream_status.append(temp)
        if isinstance(temp,type(None)):
            stream.append(x_test_mal[i])
            cons_as_mal+=1
        elif temp == 'cons_as_ben':
            stream.append(x_test_mal[i])
            cons_as_ben+=1
        elif temp[0]=='split':
            fooled+=1
            for pkt in temp[1]:
                p2 = pkt[0,-1]
                stream.append(p2)
        elif temp[0]=='inject':
            fooled+=1
            fake_pkt = temp[1]
            stream.append(fake_pkt)
            stream.append(x_test_mal[i])
    print ('duration:',time.time() - begin_time)


    print ("TPR in adversarial setting for "+attack_type+" is {0:0.4f}".format(cons_as_mal/len(x_test_mal)))
