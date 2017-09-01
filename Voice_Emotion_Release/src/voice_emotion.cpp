#include "voice_emotion.h"
#include"opensmilecomp.h"
#include<iostream>
#include<QEventLoop>
#include<QTimer>
#include<QDir>
#include<QFile>
#include<QLockFile>
#include<QDebug>
#include"clearfolders.h"
#include<QString>
#include<fstream>
using std::ifstream;
using std::cout;
using std::endl;
int cSER_Engine::voice_emo_recog( cEmoResult &  emo_result,const string wav_path)
{
    emo_result.reset();
    QString filepath=QString::fromStdString(wav_path);
    QString filename= filepath.mid(filepath.lastIndexOf("/")+1);
    filename = filename.left(filename.lastIndexOf(".wav"));
    cout<<"filename="<<filename.toStdString()<<endl;

    emo_result.wav_name=filename.toStdString();

    QString fea_path=feature_path+"/"+filename+".csv";


    //feature extraction

    QString cmd=openSmile_path+"/bin/SMILExtractPA -l 0 -C "+openSmile_path+"/config/gemaps/eGeMAPSv01a.conf"+\
            " -I "+filepath+" -csvoutput "+fea_path;
    //cout<<"feature extarct cmd:"<<cmd.toStdString()<<endl;

    cout<<"fea extract cmd: "<<cmd.toStdString()<<endl;
    pro->execute(cmd);


    //tensorflow

    QString result_path=result_dir+"/"+filename+".predict";
    cout<<"result_path="<<result_path.toStdString()<<endl;

    int result;

    cmd = "predict=tfm.run_lstm(sess,\""+fea_path+"\",\""+result_path+"\")";
    result = PyRun_SimpleString(cmd.toStdString().c_str());
    if(result!=0)
    {
        cout<<"emotion recognition failed."<<endl;
        QFile::remove(fea_path);
        QFile::remove(result_path);
        return -1;
    }

    ifstream fin(result_path.toStdString());
    string predict;
    string prob;
    getline(fin,predict);
    getline(fin,prob);
    fin.close();
    QStringList probability=QString::fromStdString(prob).split(" ");
    probability.removeAt(3);
    emo_result.emo_label=predict;
    for(int i=0;i<probability.size();i++)
    {
        emo_result.emo_prob[i]=atof(probability.at(i).toStdString().c_str());
    }

    QFile::remove(fea_path);
    QFile::remove(result_path);

    return 0;
}


cSER_Engine::cSER_Engine(QString work_dir)//work_dir目录必须是全路径，且目录下面有SER_Engine文件夹
{


    openSmile_runnig=false;
    tensorFlow_runnig=false;
    engine_running=false;
    engine_exstart=false;

    tensorFlow=new cTensorFlowComp(work_dir);
    openSmile=new cOpenSmileComp(work_dir);
    pro=new QProcess;

    set_workdir(work_dir);

    connect(tensorFlow,SIGNAL(started(QString,bool)),this,SLOT(state_recv(QString,bool)));
/*
    connect(tensorFlow,SIGNAL(recogition_complete(QString)),this,SIGNAL(recognition_complete(QString)));
    connect(tensorFlow,SIGNAL(stopped(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(tensorFlow,SIGNAL(out_predict_result(QString,QString,QStringList)),this,SIGNAL(out_predict_result(QString,QString,QStringList)));
    connect(tensorFlow,SIGNAL(out_predict_result(QString,QString,QStringList)),this,SLOT(send_predict_result(QString,QString,QStringList)));

    connect(openSmile,SIGNAL(recorder_started(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(openSmile,SIGNAL(recorder_stopped(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(openSmile,SIGNAL(fea_extra_finished(QString)),tensorFlow,SLOT(run_trial(QString)));

    connect(this,SIGNAL(stop_all()),tensorFlow,SLOT(stop_tf()));
    connect(this,SIGNAL(stop_all()),openSmile,SLOT(stop_recorder()));
    connect(this,SIGNAL(fea_extact(QString)),openSmile,SLOT(fea_extract(QString)));

*/


    Py_Initialize();


}
void cEmoResult::reset()
{
    wav_name="";
    emo_label="";
    for(int i=0;i<4;i++)
    {
        emo_prob[i]=0;
    }
}

cEmoResult::cEmoResult()
{
    reset();
}
int cSER_Engine::initiate()
{
    if(engine_exstart)
    {
        cout<<"SER_Engine pre-boot failed.Engine is already in exstart state."<<endl;
        return -1;
    }

    QDir dir1,dir2;
    dir1.setPath(wav_seg_path);
    dir2.setPath(feature_path);
    dir1.setFilter(QDir::Files );
    dir2.setFilter(QDir::Files );
    QFileInfoList list1,list2;
    int count;
    count=0;
    list1=dir1.entryInfoList();
    list2=dir2.entryInfoList();

    if(list1.size()!=0||list2.size()!=0)
    {
        //清空wav_seg_path和feature_path中文件
        cout<<"clearing tempdata..."<<endl;
        while(count<10 && (list1.size()!=0||list2.size()!=0))
        {
            removeFolderContent(wav_seg_path);
            removeFolderContent(feature_path);
            QEventLoop eventloop;
            QTimer::singleShot(1000, &eventloop, SLOT(quit())); //wait 2s
            eventloop.exec();
            dir1.refresh();
            dir2.refresh();
            list1=dir1.entryInfoList();
            list2=dir2.entryInfoList();
            count++;
        }

        if(list1.size()!=0||list2.size()!=0)
        {
            cout<<"clear tempdata failed."<<endl;
            cout<<"SER Engine start failed"<<endl;
            return -1;
        }
        cout<<"tempdata cleared."<<endl;
    }

    cout<<"\nSER Engine prestrating..."<<endl;

    //开启 tensorflow module
    cout<<"starting tensorflow"<<endl;

    count=0;
    tensorFlow->start();

    while(count<5 && !tensorFlow_runnig)
    {
        QEventLoop eventloop;
        QTimer::singleShot(2000, &eventloop, SLOT(quit())); //wait 2s
        eventloop.exec();
        count++;
    }

    if(!tensorFlow_runnig)
    {
        cout<<"tensorFlow start failed."<<endl;
        cout<<"SER Engine pre-boot failed"<<endl;

        return -1;
    }



   cout<<"SER Engine is in exstart state."<<endl;
   engine_exstart=true;
    //QMessageBox::about(NULL, "About", "SER Engine started.");

    return 0;
}






int cSER_Engine::state_recv(QString compName,bool state)
{
    //cout<<"This is state_recv speaking:compName="<<compName.toStdString()<<",state="<<state<<endl;
if (compName=="openSmile")
    {
        openSmile_runnig=state;
    }
    else if(compName=="tfModule")
    {
        tensorFlow_runnig=state;
    }

    return 0;
}



int cSER_Engine::set_workdir(QString work_dir)//设置工作目录
{
    this->work_dir=work_dir;
    engine_path=work_dir+"/SER_Engine";
    model_path=engine_path+"/model";
    tempdata_path=engine_path+"/tempdata";
    wav_seg_path=tempdata_path+"/wav_seg";
    feature_path=tempdata_path+"/feature";
    openSmile_path=engine_path+"/opensmile";
    result_dir=tempdata_path+"/result";
    if(tensorFlow!=NULL)
    {
        tensorFlow->set_workdir(work_dir);
    }
    if(openSmile!=NULL)
    {
       // cout<<"work_dir="<<work_dir.toStdString()<<endl;
        openSmile->set_workdir(work_dir);
    }

    return 0;
}

 cSER_Engine::~cSER_Engine()
{

    if(pro)
    {
        pro->close();
        delete pro;
    }
    delete openSmile;
    delete tensorFlow;
    Py_FinalizeEx();

}


