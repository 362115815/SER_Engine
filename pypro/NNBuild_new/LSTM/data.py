import tensorflow as tf
import os

from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def read_feature_file(filename_queue):
	reader = tf.WholeFileReader()
	key, value = reader.read(filename_queue)
	

	return feature,label
def input_pipeline(filelists,batch_size,read_threads,is_shuffle,num_epochs):
	'''


	filenames=[]
	for filelist in filelists:
		with open(filelist,"r") as fin:
			filenames.extend(fin.readlines())

	filename_queue=tf.train.string_input_producer(filenames,num_epochs=num_epochs,shuffle=is_shuffle)

	example_list = [read_csv_file(filename_queue) for _ in range(read_threads)]



	min_after_dequeue = 5000
	capacity = min_after_dequeue + (read_threads+2)*batch_size
	example_batch, label_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

	return example_batch, label_batch
	'''
	filenames=['test.csv']
	'''
	for filelist in filelists:
		with open(filelist,"r") as fin:
			filenames.extend(fin.readlines())
	'''
	filename_queue = tf.train.string_input_producer(
    filenames, num_epochs=num_epochs, shuffle=True)
	example, label = read_csv_file(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size
	example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
	return example_batch, label_batch

with tf.Session() as sess:
	#coord=tf.train.Coordinator()
	#threads=tf.train.start_queue_runners(coord=coord)
	tf.train.start_queue_runners()
	example_batch,label_batch=input_pipeline(train_data_path,batch_size,read_threads,is_shuffle,num_epochs)
	examplels,labes=sess.run([example_batch,label_batch])
	print(examples.shape)