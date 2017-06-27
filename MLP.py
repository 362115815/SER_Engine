
import numpy as np
import random
import tensorflow as tf

class CDataSet():
	def __init__(self,data,labels,shuffle=False):
		if len(data)==0 or len(data)!=len(label)
			raise ValueError('data为空或data与label长度不匹配')
		self.data=data
		self.labels=labels
		self.batch_id = 0
		self.is_shuffle=shuffle

	def _shuffle(self):
		c=list(zip(self.data,self.labels))
		random.shuffle(c)
		self.data[:],self.labels[:]=zip(*c)

    def next_batch(self, batch_size): #如果到达末尾，则把batch_size返回0，否则返回所读取的batch_size
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
            return [],[],0

        if(self.batch_id==0):
        	if is_shuffle==True:
        		_shuffle()
            		
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])

        self.batch_id = min(self.batch_id + batch_size, len(self.data))

        return batch_data, batch_labels,batch_size


rootdir='D:/xiaomin/'
feadir=rootdir + '/feature'
emo_classes={'anger':0,'elation':1,'neutral':2,'panic':3,'sadness':4}

data_set=[]
set_num=10
fea_dim=88
class_num=len(emo_classes)

for i in range(set_num):
	filepath=feadir+'/'+str(i)+'.txt'
	with open(filepath,'r') as fin:
		data_set.append(fin.readlines())



#CV

for i in range(set_num):

	#create val_set,val_label,train_set,train_label

	val_set=[]
	val_labels=[]
	train_set=[]
	train_labels=[]	

	for j in range(set_num):

		for item in data_set[j]:
			temp = item.split(',')
			fea=float(temp[1:-1])
			label=np.zeros(len(class_num))
			label[emo_classes[temp[-1].replace('\n')]]=1

			if j==i:	
				val_set.append(fea)
				val_labels.append(label)
			else:
				train_set.append(fea)
				train_labels.append(label)

	#start model training

	val_set=np.array(val_set)
	val_labels=np.array(val_labels)
	train_set=np.array(train_set)
	train_labels=np.array(train_labels)

	epoch_num=10
	batch_size=50
	learning_rate=0.001	
	hidden_layer=[512]

	hidden_layer_num= len(hidden_layer)
	in_node_num=fea_dim
	out_node_num=class_num

	W=[]
	b=[]



	#draw neural net
	x=tf.placeholder('float',[None,fea_dim])
	y_=tf.placeholder('float',[None, fea_dim])

	net_size=[]
	net_size.append(in_node_num)

	net_size.append(item for item in hidden_layer)

	net_size.append(out_node_num)

	layer_size=len(net_size)-1

	for index in range(1,len(net_size)):
		W.append(tf.Variable(tf.random_normal([net_size[index-1],net_size[index]])))
		b.append(tf.Variable(tf.random_normal([net_size[index]])))


	layer_out=range(net_size)

	layer_out[0]=x

	for index in range(1,net_size-1):
		layer_out[index]=tf.add(tf.matmul(layer_out[index-1],W[index-1]),b[index-1])
		layer_out[index]=tf.nn.relu(layer_out[index])

	layer_out[-1]=tf.add(tf.matmul(layer_out[-2],W[-1]),b[-1])
	y=tf.nn.softmax(layer_out[-1])

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

	init=tf.global_variables_initialize()

	correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

	#normalize
	# z-score
	mu=np.average(train_set)
	sigma=np.std(train_set)


	train_set=(train_set-mu)/sigma
	val_set=(val_set-mu)/sigma

	#run epoch
	train_data=CDataSet(train_set,train_labels)

	with tf.Session() as sess:

		sess.run(init)

		avg_cost = np.zeros(epoch_num)
		acc_val = np.zeros(epoch_num)
		acc_train = np.zeros(epoch_num)

		for epoch in range(epoch_num):
			batch_num=0
			while True:
				batch = train_data.next_batch(batch_size)
				if batch[2]==0:
					break

				_, c=sess.run([optimizer, cost],feed_dict={x:batch[0],y_:batch[1]})

				batch_num=batch_num+1
				avg_cost[epoch]= (batch_num-1)/batch_num*avg_cost[epoch]+c/batch_num

			acc_val[epoch]=accuracy.eval({x:val_set,y_:val_labels})
			acc_train[epoch]=accuracy.eval({x:train_set,y_:train_labels})
			






		















