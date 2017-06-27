from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
 
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_to(data_set, despath):
  """Converts a dataset to tfrecords."""
 	row_num,col_num=np.shape(data_set)
 	writer = tf.python_io.TFRecordWriter(despath)

 	for item in data_set:
 		item = item.split(",")
 		filename = item[0]
 		label = item[-1].replace('\n','')
 		fea_data=item[1:-1]
 		example=tf.train.Example(features=tf.train.Features(feature={
 			'label': _bytes_feature(label),
 			'filename':_bytes_feature(filename),
 			'fea_dim':_int64_feature(col_num-2),
 			'fea_data':_bytes_feature(fea_data)
 			}
 			))

 		writer.write(example.SerializeToString())

 	writer.close()










  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


def main(unused_argv):
	
	#read data
	with open(FLAGS.file,'r') as fin:
		data_set =  fin.readlines()
		convert_to(data_set,FLAGS.despath)



if __name__ == '__main__':
 	parser = argparse.ArgumentParser()
	  parser.add_argument(
      '--filepath',
      type=str,
      help='Path to load the origin data file'
  )
	  parser.add_argument(
      '--despath',
      type=str,
      help='Path to write the tfrecord data converted from the origin csv data'
  )	  
	
	FLAGS,unparsed = parser.parse_known_args()
	tf.app.run(main = main, argv=[sys.argv[0]]+ unparsed)





