root_dir = "D:/xiaomin"
data_dir = root_dir+" /feature"

file_num = 10
fea_data = [] 

for i in range(file_num):
	filepath = data_dir+"/"+str(i)+".txt"
	fea_data.append(open(filepath))



