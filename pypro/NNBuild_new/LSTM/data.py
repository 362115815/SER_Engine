import tensorflow as tf
import os
from tensorflow.python import debug as tfdbg
from config import *
import time

os.system("clear")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def read_csv_file(filename_queue):
	reader = tf.WholeFileReader()
	key, value = reader.read(filename_queue)
	feature=tf.string_split([value],delimiter="\n")
	feature=tf.string_split(feature.values,delimiter=";")
	fea_dense=tf.sparse_tensor_to_dense(feature,default_value="#") 
	fea_ =fea_dense[:,2:]
	label=fea_dense[0,0]
	label=tf.string_split([label],delimiter="\'")
	fea_final=tf.string_to_number(fea_)
	label=tf.string_to_number(label.values,out_type=tf.int32)
	label_final=tf.one_hot(label,class_num,dtype=tf.float32)
	length= tf.shape(fea_final)[0]
	return key,fea_final,label_final,length

def input_pipeline(filelists, batch_size, num_epochs,shuffle):
	filenames=[]
	for filelist in filelists:
		with open(filelist,"r") as fin:
			temp=fin.readlines()
			for item in temp:
				filenames.append(item.strip())

	
	filename_queue = tf.train.string_input_producer(filenames, seed=time.time(),num_epochs=num_epochs, shuffle=shuffle)

	key,example, label,length= read_csv_file(filename_queue)

	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size
	key_batch,example_batch, label_batch,legnth_batch = tf.train.batch([key,example, label,length], batch_size=batch_size,dynamic_pad=True,
		allow_smaller_final_batch=True,capacity=capacity)

	return key_batch,example_batch, label_batch,legnth_batch



key_batch,example_batch,label_batch,legnth_batch=input_pipeline(train_data_path,batch_size,num_epochs,shuffle)



with tf.Session() as sess:

	#tf.global_variables_initializer().run()

	tf.local_variables_initializer().run()

	#tf.initialize_all_variables().run()

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	try:
		while not coord.should_stop():
			key_batch1,feature1,label1,legnth_batch1 = sess.run([key_batch,example_batch,label_batch,legnth_batch])
			print(feature1)
			print(label1)
			print(key_batch1)
			print(legnth_batch1)
	except tf.errors.OutOfRangeError:
		print 'Done training -- epoch limit reached'
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

	coord.join(threads)

