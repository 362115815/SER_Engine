#include "opensmilecomp.h"
#include<iostream>
#include <windows.h>
#include<string>
#include<cstdio>
#include<QEventLoop>
#include<QTimer>
#include<QDebug>
#include<QFile>
using std::cout;
using std::endl;
using std::string;
cOpenSmileComp::cOpenSmileComp(QString workdir)
{
    setWorkdir(workdir);
    pro=NULL;
    console_msg="";
    return;

}

 void cOpenSmileComp::setWorkdir(QString workdir)
 {
     this->workdir=workdir;
     engine_path=workdir+"/SER_Engine";
     opensmile_path=engine_path+"/opensmile";
     tempdata_path=engine_path+"/tempdata";
     wav_seg_path=tempdata_path+"/wav_seg";
     feature_path=tempdata_path+"/feature";
     return;
 }

int cOpenSmileComp::start_record(QString workdir)
{
    setWorkdir(workdir);
    QString cmd=this->opensmile_path+"/bin/SMILExtractPA -C "+opensmile_path+"/config/myconfig/live_rec.conf"+\
            " -filebase "+wav_seg_path+"/output_segment_";


    pro =new QProcess(this);
    connect(pro,SIGNAL(readyReadStandardError()),this,SLOT(readOutput()));
    connect(pro,SIGNAL(readyRead()),this,SLOT(readOutput()));
    setvbuf(stdout, (char *)NULL, _IONBF, 0);//关闭缓冲
    pro->setProcessChannelMode(QProcess::MergedChannels);

    pro->start(cmd);
    int count=0;

    while(count<5)
    {
       if(console_msg.indexOf("Pa_StartStream: waveInStart returned = 0x0.")!=-1)
       {

           emit recorder_started("openSmile",true);
           cout<<"recorder started."<<endl;
           return 0;
       }
       count++;
       QEventLoop eventloop;
       QTimer::singleShot(2000, &eventloop, SLOT(quit())); //wait 2s
       eventloop.exec();
    }

    cout<<"recorder start failed."<<endl;
    return -1;

}
int cOpenSmileComp::stop_record()
{
    cout<<"stopping recorder..."<<endl;
    if(pro)
    {
        pro->close();
       // string msg=pro->readAll().toStdString();
       // cout<<msg<<endl;
        cout<<"recorder stopped."<<endl;
        emit recorder_stopped("openSmile",false);
    }
    return 0;
}

cOpenSmileComp::~cOpenSmileComp()
{
    if(pro)
    {
        delete pro;
    }
}
void cOpenSmileComp::readOutput()//读取输出
{
    console_msg=pro->readAll();
    cout<<console_msg.toStdString();
    return;
}
void cOpenSmileComp::stop_recorder()//停止录音
{
    stop_record();
}
int cOpenSmileComp::fea_extract(QString wavfile)//提取特征
{



    qDebug()<<"this is fea_extarct speaking:"<<endl;
    wav_queue.enqueue(wavfile);
    qDebug()<<"wavefile:"<<wavfile;
    while(!wav_queue.isEmpty())
    {
        QString wavname=wav_queue.dequeue();
        //调用opensmile 提取特征
        //qDebug()<<"wavname:"<<wavname;
         //qDebug()<<"feature_path:"<<feature_path;
        QString filename= wavname.mid(wavname.lastIndexOf("/"));
        //qDebug()<<"feaname:"<<feaname;
        filename = feature_path+filename.left(filename.lastIndexOf("_done.wav")).append(".csv");
        //qDebug()<<"feaname:"<<feaname;
        QFile file;
        file.setFileName(filename);

        QString cmd=opensmile_path+"/bin/SMILExtractPA -C "+opensmile_path+"/config/gemaps/eGeMAPSv01a.conf"+
                " -I "+wavname+" -csvoutput "+filename;

        QProcess pro(this);
        pro.execute(cmd);


        int count=0;
        bool rt=false;
        while(count<100)
        {
            //qDebug()<<filename.left(filename.lastIndexOf(".wav"))+"_done.wav";
            rt =file.rename(filename.left(filename.lastIndexOf(".csv"))+"_done.csv");
            if(rt)
            {
                //qDebug()<<file.fileName()<<endl;
                emit fea_extra_finished(wavname,file.fileName());
                break;
            }
            count++;
            QEventLoop eventloop;
            QTimer::singleShot(50, &eventloop, SLOT(quit()));
            eventloop.exec();
        }
        cout<<rt<<endl;
        cout<<count<<endl;
        if(!rt)
        {
            cout<<"fea_extract Error: file can not be renamed."<<endl;
            return -1;
        }





    }




    return 0;
}
