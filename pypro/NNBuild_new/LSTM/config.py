


# data set config 

train_data_path=["/data/mm0105.chen/wjhan/xiaomin/feature/iemo/lld/filelist.txt"]
eval_data_path=[]

shuffle = True

read_threads= 1

class_num = 4

batch_size = 256

num_epochs=10


# training config

stop_presition=0.98 # training will stop if epoch reaches epoch_num or accuracy on training set 

learning_rate = 0.001

do_dropout  =  False

do_batchnorm = False



