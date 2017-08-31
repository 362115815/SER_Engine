'''merge arff feature to a single file '''
import data_reader as dr

out_file_path="D:/xiaomin/feature/record_openInn_NOISE_feature/intern_noise_all.arff"
filelist_path="D:/xiaomin/feature/record_openInn_NOISE_feature/filelist.txt"

with open(filelist_path) as fin:
    filelist=fin.readlines()

with open(out_file_path,'w') as fout:    
    for file in filelist:
        #(file)
        fea=dr.ArffReader(file.strip())
        for item in fea.data:
            fout.write(item)





