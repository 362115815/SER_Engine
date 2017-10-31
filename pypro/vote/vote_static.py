import os
import numpy as np 
np.set_printoptions(threshold=np.inf) 








root_dir ='/data/mm0105.chen/wjhan/dzy/LSTM'
label_path="/data/mm0105.chen/wjhan/dzy/LSTM/votetest1/test_labels.txt"

cv_num=84

lab_no='votetest_shuffle0.6'

count=0

with open(os.path.join(root_dir,lab_no,"filelist.txt"),'r') as fin:
	cv_list=fin.readlines()

predicts=[]


#read label
#
#[0, 0, 0, 1, 'ENUS_M_007_env00_001_sad']

with open(label_path,'r') as fin:
	temp=fin.readlines()
	labels=[]
	for label in temp:
		label=label.strip().strip("[")
		label=label.split(",")[:4]
		label=map(eval,label)
		label= np.argmax(label)
		#print(label)
		labels.append(label)
#print(labels)



cv_dict={}

for cv_path in cv_list:
	predicts_path=os.path.join(cv_path.strip(),"predict.txt")
	if not os.path.exists(predicts_path):
		print("file not exitst:%s"%predicts_path)
		continue 
	cv_num=cv_path.strip().split("_")[-1]
	cv_num=int(cv_num[2:])
	cv_dict[cv_num]=count
	count=count+1
	with open(os.path.join(cv_path.strip(),"predict.txt"),"r") as fin:
		temp=fin.readlines()
		cv_rt=[]
		for rt in temp:
			rt=rt.strip().strip("]").strip("[").strip()
			rt=rt.split()
			#print(rt)
			cv_rt.append(map(eval, rt))
		#print(cv_rt)
		#print(np.shape(cv_rt))
	#print(np.shape(cv_rt))
	predicts.append(cv_rt)
	#print(np.shape(predicts))

predicts=np.array(predicts)

#print(cv_dict)

#print(np.shape(predicts[0]))


each_cv_acc=[]

acc_max=-1
acc_max_cv=-1
acc_max_rt=[]
for i in range(count):
	predict=predicts[i]
	#print(np.shape(predict))
	rt=np.argmax(predict,axis=1)
	#print(predict[0:10,:])
	#print(rt[0:10])
	#print(np.shape(rt))
	right_num=np.equal(rt,labels)
	total_num=len(right_num)

	right_num=sum(right_num)

	#print(right_num)
	#print(total_num)
	accuracy=float(right_num)/total_num
	#print(accuracy)
	if acc_max<accuracy:
		acc_max=accuracy
		acc_max_cv=i
		acc_max_rt=rt


	each_cv_acc.append(accuracy)

	#print(right_num)

	#print(len(right_num))

	#print(np.shape(right_num))

	#print(sum(right_num))

#print("acc_max:%f"%acc_max)
#print("acc_max_cv:%d"%acc_max_cv)
print("\n")
print("==================== ============== ===================")
print("==================== Model from All ===================")
print("==================== ============== ===================")

print('\n')
##1
print('===============Summary================')

print("\n")
print("max_cv:")
print('\n')

print("acc_average:%f"%np.average(each_cv_acc))
print("acc_max:%f"%acc_max)
print("acc_min:%f"%np.min(each_cv_acc))
print("acc_max_cv:%d"%acc_max_cv)


#vote
print('\n')
print("by_accuracy:")
print('\n')
cv_pred_sum=np.sum(predicts,axis=0)
#print(cv_pred_sum)
#print(np.shape(cv_pred_sum))

cv_vote_rt=np.argmax(cv_pred_sum,axis=1)
#print(cv_vote_rt)
#print(np.shape(cv_vote_rt))

right_num=np.equal(cv_vote_rt,labels)
total_num=len(right_num)

right_num=sum(right_num)

print("right_num:%d"%right_num)
print("total_num:%d"%total_num)


accuracy=float(right_num)/total_num
print("accuracy after vote(by accuracy):%f"%accuracy)

right_num=np.equal(acc_max_rt,cv_vote_rt)
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))




temp=np.argmax(predicts,axis=2)
#print(temp)
#print(np.shape(temp))

temp=np.transpose(temp)
#print(np.shape(temp))
#print(temp[1])




print('\n')
print("by_argmax:")
print('\n')
cv_vote_rt_1=[]

for line in temp:
	rt=np.argmax(np.bincount(line))
	cv_vote_rt_1.append(rt)


right_num=np.equal(cv_vote_rt_1,labels)
total_num=len(right_num)

right_num=sum(right_num)

print("right_num:%d"%right_num)
print("total_num:%d"%total_num)

accuracy=float(right_num)/total_num


print("accuracy after vote(by argmax):%f"%accuracy)

right_num=np.equal(acc_max_rt,cv_vote_rt_1)
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))



#1
print('\n')
print('================ENUS===================')
print('\n')
print('-------------Meeting Room--------------')

begin=0
end=1017

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


#print(np.shape(right_num))



print('\n')

print('----------------Office-----------------')


begin=1017
end=1544

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))





print('\n')
print('------------Shopping Mall--------------')
print('\n')


begin=1544
end=2283

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))






#1
print('\n')
print('================Korea===================')
print('\n')
print('-------------Meeting Room--------------')

begin=2283
end=2808

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


#print(np.shape(right_num))



print('\n')

print('----------------Office-----------------')


begin=2808
end=3473

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))




print('\n')

print('------------Shopping Mall--------------')
print('\n')


begin=3473
end=4233

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')

print('\n')
print("============ ======================= ================")
print("============ Model from Meeting Room ================")
print("============ ======================= ================")

print('\n')

CV_M=[1,3,4,5,9,10,11,12,13,15,16,18,19,20,29,30,31,32,33,34,46,47,48,49,50,57,58,59,60,61,72,73,74,75,84,85,86,87,88]



print('\n')
predicts_M=[]
for index in CV_M:
	if index not in cv_dict.keys():
		print("cv%d doesn't in prediction set"%index)
		continue
	n_idx=cv_dict[index]
	predicts_M.append(predicts[n_idx])

print(np.shape(predicts_M))

Model_num=len(predicts_M)

print("model_num:%d"%Model_num)

each_cv_acc=[]

acc_max=-1
acc_max_cv=-1
acc_max_rt=[]
for i in range(Model_num):
	predict=predicts_M[i]
	#print(np.shape(predict))
	rt=np.argmax(predict,axis=1)
	#print(predict[0:10,:])
	#print(rt[0:10])
	#print(np.shape(rt))
	right_num=np.equal(rt,labels)
	total_num=len(right_num)

	right_num=sum(right_num)

	#print(right_num)
	#print(total_num)
	accuracy=float(right_num)/total_num
	#print(accuracy)
	if acc_max<accuracy:
		acc_max=accuracy
		acc_max_cv=i
		acc_max_rt=rt

	each_cv_acc.append(accuracy)

print("\n")

#M
print('===============Summary================')

print("\n")
print("max_cv:")
print('\n')

print("acc_average:%f"%np.average(each_cv_acc))
print("acc_max:%f"%acc_max)
print("acc_min:%f"%np.min(each_cv_acc))
print("acc_max_cv:%d"%acc_max_cv)


#vote
print('\n')
print("by_accuracy:")
print('\n')
cv_pred_sum=np.sum(predicts_M,axis=0)
#print(cv_pred_sum)
#print(np.shape(cv_pred_sum))

cv_vote_rt=np.argmax(cv_pred_sum,axis=1)
#print(cv_vote_rt)
#print(np.shape(cv_vote_rt))

right_num=np.equal(cv_vote_rt,labels)
total_num=len(right_num)

right_num=sum(right_num)

print("right_num:%d"%right_num)
print("total_num:%d"%total_num)


accuracy=float(right_num)/total_num
print("accuracy after vote(by accuracy):%f"%accuracy)

right_num=np.equal(acc_max_rt,cv_vote_rt)
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))




temp=np.argmax(predicts_M,axis=2)
#print(temp)
#print(np.shape(temp))

temp=np.transpose(temp)
#print(np.shape(temp))
#print(temp[1])




print('\n')
print("by_argmax:")
print('\n')
cv_vote_rt_1=[]

for line in temp:
	rt=np.argmax(np.bincount(line))
	cv_vote_rt_1.append(rt)


right_num=np.equal(cv_vote_rt_1,labels)
total_num=len(right_num)

right_num=sum(right_num)

print("right_num:%d"%right_num)
print("total_num:%d"%total_num)

accuracy=float(right_num)/total_num


print("accuracy after vote(by argmax):%f"%accuracy)

right_num=np.equal(acc_max_rt,cv_vote_rt_1)
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))




#M
print('\n')
print('================ENUS===================')
print('\n')
print('-------------Meeting Room--------------')

begin=0
end=1017

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


#print(np.shape(right_num))



print('\n')

print('----------------Office-----------------')


begin=1017
end=1544

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))





print('\n')
print('------------Shopping Mall--------------')
print('\n')


begin=1544
end=2283

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))






#M
print('\n')
print('================Korea===================')
print('\n')
print('-------------Meeting Room--------------')

begin=2283
end=2808

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


#print(np.shape(right_num))



print('\n')

print('----------------Office-----------------')


begin=2808
end=3473

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))




print('\n')

print('------------Shopping Mall--------------')
print('\n')


begin=3473
end=4233

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')





print('\n')
print("================ ================= ==================")
print("================ Model from Office ==================")
print("================ ================= ==================")

print('\n')

CV_O=[21,22,23,24,25,26,35,36,37,38,39,40,51,52,53,54,62,63,64,65,66,76,77,78,79,89,90,91,92,93]



print('\n')
predicts_O=[]
for index in CV_O:
	if index not in cv_dict.keys():
		print("cv%d doesn't in prediction set"%index)
		continue
	n_idx=cv_dict[index]
	predicts_O.append(predicts[n_idx])

print(np.shape(predicts_O))

Model_num=len(predicts_O)

print("model_num:%d"%Model_num)

each_cv_acc=[]

acc_max=-1
acc_max_cv=-1
acc_max_rt=[]
for i in range(Model_num):
	predict=predicts_O[i]
	#print(np.shape(predict))
	rt=np.argmax(predict,axis=1)
	#print(predict[0:10,:])
	#print(rt[0:10])
	#print(np.shape(rt))
	right_num=np.equal(rt,labels)
	total_num=len(right_num)

	right_num=sum(right_num)

	#print(right_num)
	#print(total_num)
	accuracy=float(right_num)/total_num
	#print(accuracy)
	if acc_max<accuracy:
		acc_max=accuracy
		acc_max_cv=i
		acc_max_rt=rt

	each_cv_acc.append(accuracy)

print("\n")

#O
print('===============Summary================')

print("\n")
print("max_cv:")
print('\n')

print("acc_average:%f"%np.average(each_cv_acc))
print("acc_max:%f"%acc_max)
print("acc_min:%f"%np.min(each_cv_acc))
print("acc_max_cv:%d"%acc_max_cv)


#vote
print('\n')
print("by_accuracy:")
print('\n')
cv_pred_sum=np.sum(predicts_O,axis=0)
#print(cv_pred_sum)
#print(np.shape(cv_pred_sum))

cv_vote_rt=np.argmax(cv_pred_sum,axis=1)
#print(cv_vote_rt)
#print(np.shape(cv_vote_rt))

right_num=np.equal(cv_vote_rt,labels)
total_num=len(right_num)

right_num=sum(right_num)

print("right_num:%d"%right_num)
print("total_num:%d"%total_num)


accuracy=float(right_num)/total_num
print("accuracy after vote(by accuracy):%f"%accuracy)

right_num=np.equal(acc_max_rt,cv_vote_rt)
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))




temp=np.argmax(predicts_O,axis=2)
#print(temp)
#print(np.shape(temp))

temp=np.transpose(temp)
#print(np.shape(temp))
#print(temp[1])




print('\n')
print("by_argmax:")
print('\n')
cv_vote_rt_1=[]

for line in temp:
	rt=np.argmax(np.bincount(line))
	cv_vote_rt_1.append(rt)


right_num=np.equal(cv_vote_rt_1,labels)
total_num=len(right_num)

right_num=sum(right_num)

print("right_num:%d"%right_num)
print("total_num:%d"%total_num)

accuracy=float(right_num)/total_num


print("accuracy after vote(by argmax):%f"%accuracy)

right_num=np.equal(acc_max_rt,cv_vote_rt_1)
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))




#O
print('\n')
print('================ENUS===================')
print('\n')
print('-------------Meeting Room--------------')

begin=0
end=1017

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


#print(np.shape(right_num))



print('\n')

print('----------------Office-----------------')


begin=1017
end=1544

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))





print('\n')
print('------------Shopping Mall--------------')
print('\n')


begin=1544
end=2283

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))






#O
print('\n')
print('================Korea===================')
print('\n')
print('-------------Meeting Room--------------')

begin=2283
end=2808

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


#print(np.shape(right_num))



print('\n')

print('----------------Office-----------------')


begin=2808
end=3473

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))




print('\n')

print('------------Shopping Mall--------------')
print('\n')


begin=3473
end=4233

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')




print('\n')
print("============ ======================== ===============")
print("============ Model from Shopping Mall ===============")
print("============ ======================== ===============")

print('\n')

CV_S=[27,28,41,42,43,44,45,55,56,67,68,69,70,71,80,81,82,83,94,95,96,97,98]



print('\n')
predicts_S=[]
for index in CV_S:
	if index not in cv_dict.keys():
		print("cv%d doesn't in prediction set"%index)
		continue
	n_idx=cv_dict[index]

	predicts_S.append(predicts[n_idx])

print(np.shape(predicts_S))

Model_num=len(predicts_S)

print("model_num:%d"%Model_num)

each_cv_acc=[]

acc_max=-1
acc_max_cv=-1
acc_max_rt=[]
for i in range(Model_num):
	predict=predicts_S[i]
	#print(np.shape(predict))
	rt=np.argmax(predict,axis=1)
	#print(predict[0:10,:])
	#print(rt[0:10])
	#print(np.shape(rt))
	right_num=np.equal(rt,labels)
	total_num=len(right_num)

	right_num=sum(right_num)

	#print(right_num)
	#print(total_num)
	accuracy=float(right_num)/total_num
	#print(accuracy)
	if acc_max<accuracy:
		acc_max=accuracy
		acc_max_cv=i
		acc_max_rt=rt

	each_cv_acc.append(accuracy)

print("\n")

#S
print('===============Summary================')

print("\n")
print("max_cv:")
print('\n')

print("acc_average:%f"%np.average(each_cv_acc))
print("acc_max:%f"%acc_max)
print("acc_min:%f"%np.min(each_cv_acc))
print("acc_max_cv:%d"%acc_max_cv)


#vote
print('\n')
print("by_accuracy:")
print('\n')
cv_pred_sum=np.sum(predicts_S,axis=0)
#print(cv_pred_sum)
#print(np.shape(cv_pred_sum))

cv_vote_rt=np.argmax(cv_pred_sum,axis=1)
#print(cv_vote_rt)
#print(np.shape(cv_vote_rt))

right_num=np.equal(cv_vote_rt,labels)
total_num=len(right_num)

right_num=sum(right_num)

print("right_num:%d"%right_num)
print("total_num:%d"%total_num)


accuracy=float(right_num)/total_num
print("accuracy after vote(by accuracy):%f"%accuracy)

right_num=np.equal(acc_max_rt,cv_vote_rt)
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))




temp=np.argmax(predicts_S,axis=2)
#print(temp)
#print(np.shape(temp))

temp=np.transpose(temp)
#print(np.shape(temp))
#print(temp[1])




print('\n')
print("by_argmax:")
print('\n')
cv_vote_rt_1=[]

for line in temp:
	rt=np.argmax(np.bincount(line))
	cv_vote_rt_1.append(rt)


right_num=np.equal(cv_vote_rt_1,labels)
total_num=len(right_num)

right_num=sum(right_num)

print("right_num:%d"%right_num)
print("total_num:%d"%total_num)

accuracy=float(right_num)/total_num


print("accuracy after vote(by argmax):%f"%accuracy)

right_num=np.equal(acc_max_rt,cv_vote_rt_1)
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))




#S
print('\n')
print('================ENUS===================')
print('\n')
print('-------------Meeting Room--------------')

begin=0
end=1017

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


#print(np.shape(right_num))



print('\n')

print('----------------Office-----------------')


begin=1017
end=1544

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))





print('\n')
print('------------Shopping Mall--------------')
print('\n')


begin=1544
end=2283

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))






#S
print('\n')
print('================Korea===================')
print('\n')
print('-------------Meeting Room--------------')

begin=2283
end=2808

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


#print(np.shape(right_num))



print('\n')

print('----------------Office-----------------')


begin=2808
end=3473

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))




print('\n')

print('------------Shopping Mall--------------')
print('\n')


begin=3473
end=4233

print("max_cv:")
print('\n')

right_num=np.equal(acc_max_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

print('\n')

print("by_accuracy:")
print('\n')
right_num=np.equal(cv_vote_rt[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')
print("by_argmax:")
print('\n')
right_num=np.equal(cv_vote_rt_1[begin:end],labels[begin:end])
total_num=len(right_num)
right_num=sum(right_num)
print("right_num:%d"%right_num)
print("total_num:%d"%total_num)
accuracy=float(right_num)/total_num
print("accuracy:%f"%accuracy)

right_num=np.equal(acc_max_rt[begin:end],cv_vote_rt_1[begin:end])
total_num=len(right_num)
right_num=sum(right_num)

print("right_num with acc_max_cv:%d"%right_num)
print("right rate with acc_max_cv:%f"%(float(right_num)/total_num))


print('\n')






print("count:%d"%count)






