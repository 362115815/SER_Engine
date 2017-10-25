import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


train_data_path=["/data/mm0105.chen/wjhan/xiaomin/feature/iemo/lld/filelist.txt"]


filenames=[]
for filelist in train_data_path:
	with open(filelist,"r") as fin:
		temp=fin.readlines()
		for item in temp:
			filenames.append(item.strip())

filename_queue=tf.train.string_input_producer(filenames,shuffle=False)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
feature=value.split('\n')[:-1]

'''ecord_defaults=[[0.1] for i in range(25)]
record_defaults[0]=['']
record_defaults[1]=['']
A = tf.decode_csv(value, record_defaults=record_defaults,field_delim=';')

label=A[0]
features=A
'''
with tf.Session() as sess:
	# Start populating the filename queue.
	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	#example, label,value,key =sess.run([features,label,value,key])
	
	value,key,example =sess.run([value,key,feature])

	#print(label)
	print(value)
	print(key)


	print(len(value.split("\n")))

	print(len(example))
	coord.request_stop()
	coord.join(threads)