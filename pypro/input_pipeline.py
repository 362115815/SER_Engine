# -*- coding: utf-8 -*-

import os
import numpy as np
#import scipy.misc
#import random
#import matplotlib.pyplot as plt
import tensorflow as tf
#import view_gpu

TOP_N = 0
CURRENT_DIR = os.getcwd()


def read_label_file(txt_file):
    
    with open(txt_file, 'r') as f:
        
        images_names = []  
        labels = []
        
        for line in f:
            
            filename, label = line.split(',')
#            filename, label = line.split()
#            filename = filename.split('.')[0].split('/')[-1] + '_0.png'
            images_names.append(filename)
            labels.append(int(label))
            
        f.close()
    return images_names, labels


#==============================================================================
# def write_label_file(txt_file, top_n = 50):
#     
#     base_names, base_labels = read_label_file(txt_file)
#     image_names = os.listdir(os.path.join(
#         CURRENT_DIR, 'data', 'CK+1308_aug_with_ACGAN', 'images'))
#     
#     label0_name_list = []
#     label1_name_list = []
#     label2_name_list = []
#     label3_name_list = []
#     label4_name_list = []
#     label5_name_list = []
#     label6_name_list = []
#     label7_name_list = []
#         
#     for names in image_names:
#         
#         if names.startswith('rlabel_0'):
#             label0_name_list.append(names)
#         elif names.startswith('rlabel_1'):
#             label1_name_list.append(names)
#         elif names.startswith('rlabel_2'):
#             label2_name_list.append(names)
#         elif names.startswith('rlabel_3'):
#             label3_name_list.append(names)
#         elif names.startswith('rlabel_4'):
#             label4_name_list.append(names)
#         elif names.startswith('rlabel_5'):
#             label5_name_list.append(names)
#         elif names.startswith('rlabel_6'):
#             label6_name_list.append(names)
#         elif names.startswith('rlabel_7'):
#             label7_name_list.append(names)
#         else:
#             pass
#         
#     label0_name_list.sort(reverse = True, 
#         key = lambda x: float(x.split('prob')[1].split('_flabel')[0]))
#     label1_name_list.sort(reverse = True, 
#         key = lambda x: float(x.split('prob')[1].split('_flabel')[0]))
#     label2_name_list.sort(reverse = True, 
#         key = lambda x: float(x.split('prob')[1].split('_flabel')[0]))
#     label3_name_list.sort(reverse = True, 
#         key = lambda x: float(x.split('prob')[1].split('_flabel')[0]))
#     label4_name_list.sort(reverse = True, 
#         key = lambda x: float(x.split('prob')[1].split('_flabel')[0]))
#     label5_name_list.sort(reverse = True, 
#         key = lambda x: float(x.split('prob')[1].split('_flabel')[0]))
#     label6_name_list.sort(reverse = True, 
#         key = lambda x: float(x.split('prob')[1].split('_flabel')[0]))
#     label7_name_list.sort(reverse = True, 
#         key = lambda x: float(x.split('prob')[1].split('_flabel')[0]))
#             
#     name_list = label0_name_list[:top_n] + label1_name_list[:top_n] + \
#                 label2_name_list[:top_n] + label3_name_list[:top_n] + \
#                 label4_name_list[:top_n] + label5_name_list[:top_n] + \
#                 label6_name_list[:top_n] + label7_name_list[:top_n]
# #    name_list = name_list + base_names
#             
#     label_list = [0] * top_n + [1] * top_n + [2] * top_n + [3] * top_n + \
#                 [4] * top_n + [5] * top_n + [6] * top_n + [7] * top_n
#                 
# #    label_list = label_list + base_labels
#     
#     txt_content = zip(name_list, label_list)
#     random.shuffle(txt_content)
#             
#     with open(os.path.join(CURRENT_DIR, 'data', 'CK+1308_aug_with_ACGAN', 
#         'CK+1308_aug_') + str(top_n) + '.txt', 'w') as f:
#             
#         for name, label in zip(base_names, base_labels):
#             f.write(name)
#             f.write(',')
#             f.write(str(label))
#             f.write('\n')
#                         
#         for name, label in txt_content:
#             if name.startswith('rlabel'):
#                 assert int(name[7]) == int(label), '%s error' % name
#             f.write(name)
#             f.write(',')
#             f.write(str(label))
#             f.write('\n')
#         
#     f.close()
#==============================================================================


def inputs(dataset_dir, im_format = 'png', batch_size = 64, 
           is_train = True, name = 'inputs'):
    '''    
    A dataset shouled be like this:
    
    data_dir
        -images_dir
            -image_name1.png
            -image_name2.png
            -...
        -label_file.txt
    '''
    images_dir = 'images'
    label_file = 'CK+1308_aug_' + str(TOP_N) + '.txt'
    
    all_images_names, all_labels = read_label_file(os.path.join(dataset_dir, 
                                                                label_file))
    
    images_names = all_images_names
    labels = all_labels
#    images_names = all_images_names[:(1308+TOP_N*8)]
#    labels = all_labels[:(1308+TOP_N*8)]
    
    images_paths = [os.path.join(
        dataset_dir, images_dir) + '/' + fp for fp in images_names]
    
    all_images = tf.convert_to_tensor(images_paths, dtype = tf.string)
    all_labels = tf.convert_to_tensor(labels, dtype = tf.int32)
    
#    images_paths = images_paths[:1000]
#    labels = labels[:1000]
#    test_num_max = [32, 14, 5, 18, 7, 21, 8, 25]
#    test_num_max = [37, 19, 10, 23, 12, 26, 13, 30]# aug_per_class_50
#    test_num_max = [43, 24, 15, 28, 17, 30, 19, 35]# aug_per_class_100
#    test_num_max = [47, 29, 20, 33, 23, 36, 23, 40]# aug_per_class_150
#    test_num_max = [53, 34, 25, 38, 28, 41, 28, 45]# aug_per_class_200
#    test_num_max = [57, 39, 30, 43, 33, 46, 33, 50]# aug_per_class_250
#    test_num_max = [62, 44, 35, 48, 38, 51, 38, 55]# aug_per_class_300
#    test_num_max = [67, 49, 40, 53, 43, 56, 43, 60]# aug_per_class_350
#    test_num_max = [72, 54, 45, 58, 48, 61, 48, 65]# aug_per_class_400
#    test_num_max = [77, 59, 50, 63, 53, 66, 53, 70]# aug_per_class_450
#    test_num_max = [82, 64, 55, 68, 58, 71, 58, 75]# aug_per_class_500
    
#    test_num_max = [5] * 8# gen_per_class_50    
#    test_num_max = [10] * 8# gen_per_class_100    
#    test_num_max = [15] * 8# gen_per_class_150    
#    test_num_max = [20] * 8# gen_per_class_200    
#    test_num_max = [25] * 8# gen_per_class_250    
#    test_num_max = [30] * 8# gen_per_class_300    
#    test_num_max = [35] * 8# gen_per_class_350    
#    test_num_max = [40] * 8# gen_per_class_400    
#    test_num_max = [45] * 8# gen_per_class_450
#    test_num_max = [50] * 8# gen_per_class_500
    
    test_num = [0] * 8
#    partitions_aug = [0] * len(labels)
#    partitions_aug = [0] * (len(labels) - 1308)
    partitions_base = np.load(os.path.join(dataset_dir, 'partitions_base.npy'))
#    partitions_base[partitions_base == 0] = 2
    partitions_base = list(partitions_base)
#    partitions_base = [0] * 1308
    
#==============================================================================
#     for idx, value in enumerate(labels[1308:]):
#         if value == 0:
#             test_num[0] += 1
#             partitions_aug[idx] = 1
#             if test_num[0] == TOP_N / 10:
#                 break
#             
#     for idx, value in enumerate(labels[1308:]):
#         if value == 1:
#             test_num[1] += 1
#             partitions_aug[idx] = 1
#             if test_num[1] == TOP_N / 10:
#                 break
#     
#     for idx, value in enumerate(labels[1308:]):
#         if value == 2:
#             test_num[2] += 1
#             partitions_aug[idx] = 1
#             if test_num[2] == TOP_N / 10:
#                 break
#     
#     for idx, value in enumerate(labels[1308:]):
#         if value == 3:
#             test_num[3] += 1
#             partitions_aug[idx] = 1
#             if test_num[3] == TOP_N / 10:
#                 break
#             
#     for idx, value in enumerate(labels[1308:]):
#         if value == 4:
#             test_num[4] += 1
#             partitions_aug[idx] = 1
#             if test_num[4] == TOP_N / 10:
#                 break
#             
#     for idx, value in enumerate(labels[1308:]):
#         if value == 5:
#             test_num[5] += 1
#             partitions_aug[idx] = 1
#             if test_num[5] == TOP_N / 10:
#                 break
#             
#     for idx, value in enumerate(labels[1308:]):
#         if value == 6:
#             test_num[6] += 1
#             partitions_aug[idx] = 1
#             if test_num[6] == TOP_N / 10:
#                 break
#             
#     for idx, value in enumerate(labels[1308:]):
#         if value == 7:
#             test_num[7] += 1
#             partitions_aug[idx] = 1
#             if test_num[7] == TOP_N / 10:
#                 break
#==============================================================================
#    partitions = partitions_aug
    partitions = partitions_base# + partitions_aug    
    train_images, test_images = tf.dynamic_partition(
        all_images, partitions, 2)  
    train_labels, test_labels = tf.dynamic_partition(
        all_labels, partitions, 2)
    
    if is_train:
        
        input_queue = tf.train.slice_input_producer(
            [train_images, train_labels], shuffle = True)
            
        file_content = tf.read_file(input_queue[0])
        
        if (im_format.lower() == 'jpg') or (im_format.lower() == 'jpeg'):
            image = tf.image.decode_jpeg(file_content, channels = 1)
            
        elif (im_format.lower() == 'png'):
            image = tf.image.decode_png(file_content, channels = 1)
            
        else:
            print('Only support format png or jpeg')            
            os._exit(0)
            
        label = input_queue[1]
        image = tf.reshape(image, [128, 128, 1])
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        
        images_batch, labels_batch = tf.train.shuffle_batch(
                        [image, label], batch_size = batch_size, 
                        num_threads = 4, capacity = 2000 + 3 * batch_size, 
                        min_after_dequeue = 2000)
        return images_batch, labels_batch
    
    else:
        
        input_queue = tf.train.slice_input_producer(
            [test_images, test_labels], shuffle = False)
            
        file_content = tf.read_file(input_queue[0])  
        if (im_format.lower() == 'jpg') or (im_format.lower() == 'jpeg'):
            image = tf.image.decode_jpeg(file_content, channels = 1)
        
        elif (im_format.lower() == 'png'):
            image = tf.image.decode_png(file_content, channels = 1)
        
        else:
            print('Only support format png or jpeg')            
            os._exit(0)
            
        label = input_queue[1]
        image = tf.reshape(image, [128, 128, 1])
        image = tf.cast(image, tf.float32)
        image = image / 255.0
#        image = tf.random_crop(image, [120, 120, 1])
#        image = tf.image.random_flip_left_right(image)
        
        images_batch, labels_batch = tf.train.batch(
            [image, label], batch_size = 130, num_threads = 1)#, 
#                        num_threads = 4, capacity = 2000 + 3 * batch_size, 
#                        min_after_dequeue = 2000)
        return images_batch, labels_batch
        

def eval_inputs(dataset_dir, im_format = 'png', batch_size = 64, 
           is_train = True, name = 'inputs'):
    '''    
    A dataset shouled be like this:

    data_dir
        -images_dir
            -image_name1.png
            -image_name2.png
            -...
    -label_file.txt
    '''

    images_paths = [dataset_dir + fn for fn in os.listdir(dataset_dir)]
    all_images = tf.convert_to_tensor(images_paths, dtype = tf.string)
			
    input_queue = tf.train.slice_input_producer(
        [all_images], shuffle = False, capacity = 1, num_epochs = 1)
    
    im_name = input_queue[0]
    file_content = tf.read_file(im_name)

    if (im_format.lower() == 'jpg') or (im_format.lower() == 'jpeg'):
        image = tf.image.decode_jpeg(file_content, channels = 1)
		
    elif (im_format.lower() == 'png'):
        image = tf.image.decode_png(file_content, channels = 1)
    else:
        print('Only support format png or jpeg')            
        os._exit(0)
		
	# label = input_queue[1]
    image = tf.reshape(image, [128, 128, 1])
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    images_batch = tf.train.batch([image], batch_size = batch_size, 
                                  capacity = 1, num_threads = 1)
		
    return images_batch, im_name

 
#==============================================================================
# if __name__ == '__main__':
#     
#     txt_file = os.path.join(
#         CURRENT_DIR, 'data', 'CK+1308_aug_with_ACGAN', 'CK+1308_aug_0.txt')
#     write_label_file(txt_file, top_n = 500)
#==============================================================================







    