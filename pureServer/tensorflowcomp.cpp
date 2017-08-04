#include "tensorflowcomp.h"
#include<iostream>
#include<fstream>
#include<QString>
#include<string>
#include<QFile>
//#include<QMessageBox>
using std::cout;
using std::endl;
using std::string;
using std::ifstream;
cTensorFlowComp::cTensorFlowComp(QString work_dir)
{
    set_workdir(work_dir);
    tf_module_imported=false;

}
void cTensorFlowComp::set_workdir(QString work_dir)
{
    this->work_dir=work_dir;
    engine_path=work_dir+"/SER_Engine";
    model_path=engine_path+"/models";
    result_path=engine_path+"/predict.txt";

}
cTensorFlowComp::~cTensorFlowComp()
{
    stop();
}

int cTensorFlowComp::start()
{
/*
    cout<<"initialting python environment..."<<endl;
    Py_Initialize();

    if(!Py_IsInitialized())
    {
        cout<<"fail to initialting python environment."<<endl;
        return -1;
    }

*/
    string cmd;
    int result;

    //导入模块
    if(!tf_module_imported)
    {
        cout<<"importing python tensorflow module..."<<endl;
        cmd = "import tf_module as tfm";
        //cmd = "import tensorflow";
        result = PyRun_SimpleString(cmd.c_str());

        cout <<"result="<<result<< endl;
        if(result!=0)
        {
            cout<<"fail to import tensorflow module."<<endl;
            return -1;
        }
        //tf_module_imported=false;
        cout<<"tf_module imported."<<endl;
    }


    cout<<"loading trained model..."<<endl;

    //载入模型
   // string model_file = "D:/xiaomin/SER_Client_Gui/SER_Engine/models/MLP.pb";
    string model_file=model_path.toStdString()+"/MLP.pb";

    cmd = "graph=tfm.load_graph(\"" + model_file + "\")";
    cout << cmd << endl;
    result= PyRun_SimpleString(cmd.c_str());
    if(result!=0)
    {
        cout<<"fail to load trained model:"<<model_file<<endl;
        return -1;
    }
    cout<<"trained model loaded."<<endl;


    //开启Session
    cout<<"starting session..."<<endl;

    cmd = "sess=tfm.start_session(graph=graph)";
    cout << cmd << endl;
    result = PyRun_SimpleString(cmd.c_str());

    if(result!=0)
    {
        cout<<"fail to start session"<<endl;
        return -1;
    }

    cout<<"session started.waiting to run..."<<endl;

    emit started("tfModule",true);

   // Py_FinalizeEx();




    return 0;
}
 void cTensorFlowComp::stop_tf()
 {
     stop();
     return;
 }
void cTensorFlowComp::run_trial(QString fea_path)
{

   // QMessageBox::about(NULL,"run run run",fea_path);
    string cmd;
    int result;
    //识别
    cout<<"trial is comming.starting  emotion recognition..."<<endl;

    cmd = "predict=tfm.run(sess,\""+fea_path.toStdString()+"\",\""+result_path.toStdString()+"\")";

   // cout<<cmd<<endl;
    result = PyRun_SimpleString(cmd.c_str());
    //cout << result << endl;

    if(result!=0)
    {
        cout<<"emotion recognition failed."<<endl;
        emit err_occured("tensorflowComp","emotion recognition failed");
        return;
    }

    emit recogition_complete(result_path);
    ifstream fin(result_path.toStdString());
    string predict;
    string prob;
    getline(fin,predict);
    getline(fin,prob);




    QStringList probability=QString::fromStdString(prob).split(" ");

    int index=fea_path.lastIndexOf("/");
    QString voice_seg_name = fea_path.mid(index+1).replace("_done.csv","");


    emit out_predict_result(voice_seg_name,QString::fromStdString(predict),probability);

    QFile::remove(fea_path);


    cout<<"emotion is being recognized."<<endl;




    return;
}

int cTensorFlowComp::stop()
{
    //关闭session
    string cmd;
    int result;
    cout<<"stopping session..."<<endl;
    cmd = "tfm.close_session(sess)";
   // cout << cmd << endl;
    result = PyRun_SimpleString(cmd.c_str());
    if(result!=0)
    {
        cout<<"fail to stop session"<<endl;
        return -1;
    }

    cout<<"session stopped."<<endl;

    cout<<"closing python environment..."<<endl;
/*
    Py_FinalizeEx();

    if(Py_IsInitialized())
    {
        cout<<"fail to close python environment."<<endl;
        return -1;
    }
*/
    emit stopped("tfModule",false);
    cout<<"python environment closed."<<endl;


    return 0;
}
