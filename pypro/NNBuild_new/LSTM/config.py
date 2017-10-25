


# data set config 

train_data_path=["/data/mm0105.chen/wjhan/xiaomin/feature/iemo/lld/filelist.txt"]
eval_data_path=[]

is_shuffle = False

read_threads= 1







# training config

num_epochs=1

stop_presition=0.98 # training will stop if epoch reaches epoch_num or accuracy on training set 

batch_size = 1

learning_rate = 0.001


do_dropout  =  False

do_batchnorm = False



