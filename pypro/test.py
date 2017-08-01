import tf_module as tfm
import os
rootdir="D:/xiaomin/feature_test/feature"
file_prefix="D:/xiaomin/feature_test/feature"
model_path="D:/xiaomin/feature_test/frozen_model_1756.pb"

graph=tfm.load_graph(model_path)

sess=tfm.start_session(graph=graph)


for _,_,filenames in os.walk(rootdir):
    for filename in filenames:
        filepath=file_prefix+"/"+filename
        predict=tfm.run(sess,filepath)
        print(*predict)
