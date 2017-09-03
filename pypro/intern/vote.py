import tf_module as tfm 
import numpy as np
import tensorflow as tf
model_dir="D:/xiaomin/pyproj/model_make/2017-08-10_14_30_51_cv3"

test_set="D:/xiaomin/feature/intern/byperson/ori/3.txt"
emo_classes = {'ang': 0, 'hap': 1, 'nor': 2, 'sad': 4,'neu':2,'exc':1}
with open(test_set) as fin:
    data=fin.readlines()

filenames=[];
labels=[];
predicts=[];
probs=[];
sess= tfm.start_session_ckpt(model_dir)


for item in data:
    temp=item.split(',')

    name=temp[0].strip('\'')
    filenames.append(name)
    labels.append(emo_classes[name.split('_')[-1]])
   # print(labels)
   # print(name)
    fea_data=temp[1:-1]
    prob=tfm.run_lstm_data(sess,fea_data)
    probs.append(prob)
    index=np.argmax(prob)
    predicts.append(index)


correct_prediction = tf.equal(labels,predicts)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

corec,acc_1=sess.run([correct_prediction,accuracy])

print('acc_1: %f'%acc_1)

num=len(labels)

print(num)



for step in range(2,19):
    print('---------------')
    print('step=%d'%step)
    map_prob={0:[],1:[],2:[],4:[]}

    correct_num=0;
    total_num=0;
    for i in range(num):
       # print(labels[i])
        #print(type(labels[i]))
        #print(probs[i])
        map_prob[labels[i]].append(probs[i][0])
        if len(map_prob[labels[i]])==step:
            total_num=total_num+1
            col=np.average(map_prob[labels[i]],axis=0)
            
            index=np.argmax(col)
            if index==labels[i]:
                correct_num=correct_num+1
            map_prob[labels[i]]=[]

    for key,value in map_prob.items():

        if len(value)!=0:
            total_num=total_num+1
            col=np.average(value,axis=0)
            index=np.argmax(col)
            if index==key:
                correct_num=correct_num+1
    print("total_num=%d"%total_num)
    print("correct_num=%d"%correct_num)

    print("acc_%d:%f"%(step,correct_num/total_num))
    print('---------------')


sess.close()