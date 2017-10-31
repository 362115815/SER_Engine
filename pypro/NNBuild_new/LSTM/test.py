import tensorflow as tf
import os
from tensorflow.python import debug as tfdbg

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class_num={"neu":0,"hap":1,"sad":2,"ang":3}

def get_train_entry():
	train_data_path=["/data/mm0105.chen/wjhan/xiaomin/feature/iemo/lld/filelist.txt"]
	filenames=[]
	for filelist in train_data_path:
		with open(filelist,"r") as fin:
			temp=fin.readlines()
			for item in temp:
				filenames.append(item.strip())

	filename_queue=tf.train.string_input_producer(filenames,shuffle=False,num_epochs=1)
	reader = tf.WholeFileReader()
	key, value = reader.read(filename_queue)
	feature=tf.string_split([value],delimiter="\n")
	feature1=tf.string_split(feature.values,delimiter=";")
	
	fea_dense=tf.sparse_tensor_to_dense(feature1,default_value="#") 
	fea_=fea_dense[:,2:]

	label=fea_dense[0,0]

	label=tf.string_split([label],delimiter="\'")

	fea_1=tf.string_to_number(fea_)
	label_1=tf.string_to_number(label.values,out_type=tf.int32)
	pt=tf.Print(feature1.values,[tf.shape(feature1.values,"dfd")])
	label_2=tf.one_hot(label_1,4,dtype=tf.float32)

	return fea_1,label_2,pt


'''ecord_defaults=[[0.1] for i in range(25)]
record_defaults[0]=['']
record_defaults[1]=['']
A = tf.decode_csv(value, record_defaults=record_defaults,field_delim=';')

label=A[0]
features=A
'''
feature,label,pt=get_train_entry()

with tf.Session() as sess:
	# Start populating the filename queue.

	#tf.global_variables_initializer().run()

	tf.local_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	
	#example, label,value,key =sess.run([features,label,value,key])
	sess=tfdbg.LocalCLIDebugWrapperSession(sess)

	feature1,label1, _ =sess.run([feature,label,pt])

	#print(type(value1))
	#print(type(key1))
	print(feature1)
	print(len(feature1))
	print(label1)
	

#	for item in feature.values:
#		print("\n"+item)
#		print(type(item))



	#print(label)
	#print(value)
	#print(key)


	#print(len(value.split("\n")))
	#print(type(value))
	#print(type(feature))
	#print(feature.values[0])
	coord.request_stop()
	coord.join(threads)