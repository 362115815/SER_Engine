#include "ser_engine.h"
#include"opensmilecomp.h"
#include<iostream>
#include<QEventLoop>
#include<QTimer>
#include<QDir>
#include<QFile>
#include<QLockFile>
#include<QDebug>
#include"clearfolders.h"
#include<QMessageBox>
using std::cout;
using std::endl;

cSER_Engine::cSER_Engine(QString work_dir)//work_dir目录必须是全路径，且目录下面有SER_Engine文件夹
{
    this->work_dir=work_dir;
    model_path=engine_path+"/model";
    tempdata_path=engine_path+"/tempdata";
    wav_seg_path=tempdata_path+"/wav_seg";
    feature_path=tempdata_path+"/feature";
    openSmile_path=engine_path+"/opensmile";
    spyThread_running=false;
    openSmile_runnig=false;
    tensorFlow_runnig=false;
    spyThread=NULL;
    openSmile=NULL;
    tensorFlow=NULL;
}


int cSER_Engine::start_engine()
{
   set_workdir("D:/xiaomin/SER_Client_Gui");

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





    //开启监视线程
    count=0;

    cout<<"\nstrating SER Engine..."<<endl;

    spyThread=new cSpyThread(wav_seg_path);
    connect(this,SIGNAL(stop_all()),spyThread,SLOT(stop()));
    connect(spyThread,SIGNAL(threadstarted(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(spyThread,SIGNAL(threadstopped(QString,bool)),this,SLOT(state_recv(QString,bool)));

    cout<<"starting spyThread..."<<endl;
    spyThread->start();

    count=0;

    while(count<5 && !spyThread_running)
    {
        QEventLoop eventloop;
        QTimer::singleShot(2000, &eventloop, SLOT(quit())); //wait 2s
        eventloop.exec();
    }




    //开启 tensorflow module

    tensorFlow=new cTensorFlowComp(work_dir);
    connect(this,SIGNAL(stop_all()),tensorFlow,SLOT(stop_tf()));
    connect(tensorFlow,SIGNAL(recogition_complete(QString,QString)),this,SIGNAL(recognition_complete(QString,QString)));
    connect(tensorFlow,SIGNAL(started(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(tensorFlow,SIGNAL(stopped(QString,bool)),this,SLOT(state_recv(QString,bool)));
    tensorFlow->start();

    //cout<<"spyThread_running:"<<spyThread_running<<endl;
    //cout<<"openSmile_runnig:"<<openSmile_runnig<<endl;



    //开启 opensmile 进程
    openSmile=new cOpenSmileComp(work_dir);
    connect(this,SIGNAL(stop_all()),openSmile,SLOT(stop_recorder()));
    connect(openSmile,SIGNAL(recorder_started(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(openSmile,SIGNAL(recorder_stopped(QString,bool)),this,SLOT(state_recv(QString,bool)));

    cout<<"starting recorder..."<<endl;

    openSmile->start_record(work_dir);//启动 opensmile recorder

    count=0;

    while(count<5 && !openSmile_runnig)
    {
        QEventLoop eventloop;
        QTimer::singleShot(2000, &eventloop, SLOT(quit())); //wait 2s
        eventloop.exec();
        count++;
    }


    connect(openSmile,SIGNAL(fea_extra_finished(QString,QString)),tensorFlow,SLOT(run_trial(QString,QString)));


    if(spyThread_running && openSmile_runnig && tensorFlow_runnig)
    {
        connect(spyThread,SIGNAL(turnComplete(QString)),openSmile,SLOT(fea_extract(QString)));
        cout<<"SER Engine started."<<endl;

        //QMessageBox::about(NULL, "About", "SER Engine started.");

		QMessageBox box;
		box.setText(QString::fromLocal8Bit("You can speak now..."));
		box.setStyleSheet("color:red");
		QFont font;
		font.setPixelSize(20);
		box.setFont(font);
		box.setFixedSize(800, 800);
		int bret = box.exec();

		emit engine_started();
    }
    else
    {
        cout<<"SER Engine start failed"<<endl;
        emit stop_all();
        return -1;
    }

    return 0;
}


int cSER_Engine::stop_engine()
{
    //发射 spyThread 退出信号
    cout<<"stopping SER Engine..."<<endl;
    emit stop_all();

   //清空wav_seg_path和feature_path中文件
   //removeFolderContent(wav_seg_path);
   //removeFolderContent(feature_path);


    int count=0;
    while(count<5 && (spyThread_running||openSmile_runnig||tensorFlow_runnig))
    {
        QEventLoop eventloop;
        QTimer::singleShot(2000, &eventloop, SLOT(quit())); //wait 2s
        eventloop.exec();
    }


    if(!spyThread_running && !openSmile_runnig && !tensorFlow_runnig)
    {
        cout<<"SER Engine stopped."<<endl;
        QMessageBox::about(NULL, "About", "SER Engine stopped.");
    }
    else
    {
        cout<<"SER Engine stop failed"<<endl;
        return -1;
    }


   return 0;
}



int cSER_Engine::state_recv(QString compName,bool state)
{
    //cout<<"This is state_recv speaking:compName="<<compName.toStdString()<<",state="<<state<<endl;
    if(compName=="spyThread")
    {
        spyThread_running=state;
    }
    else if (compName=="openSmile")
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
    if(work_dir=="")
    {
        engine_path="./SER_Engine";
    }
    else
    {
        engine_path=work_dir+"/SER_Engine";
    }
    model_path=engine_path+"/model";
    tempdata_path=engine_path+"/tempdata";
    wav_seg_path=tempdata_path+"/wav_seg";
    feature_path=tempdata_path+"/feature";
    openSmile_path=engine_path+"/opensmile";
    return 0;
}

cSER_Engine::~cSER_Engine()
{


    if(spyThread==NULL)
    {
        delete spyThread;
    }
    if(openSmile==NULL)
    {
        delete openSmile;
    }
    if(tensorFlow==NULL)
    {
        delete tensorFlow;
    }

}


void cSpyThread::run()
{
    cout<<"spyThread started."<<endl;
    emit threadstarted("spyThread",true);
    QDir dir(wav_seg_path);

    dir.setFilter(QDir::Files | QDir::Modified);
    dir.setSorting(QDir::Name);
    int filecount=0;
     QFile file;


    while(is_loop)
    {
       // cout<<"This is spyThread speaking :spyThread is running."<<endl;
       // sleep(3);
        dir.refresh();
        QFileInfoList list=dir.entryInfoList();
        int filecur_num=list.size();


        while(filecount<filecur_num)
        {
            cout<<"filecur_num:"<<filecur_num<<endl;
            cout<<"filecount:"<<filecount<<endl;
            QString filename=wav_seg_path+"/"+list.at(filecount++).fileName();
            file.setFileName(filename);
            //qDebug()<<filename<<endl;
            int count=0;
            bool rt=false;
            while(count<100)
            {
                //qDebug()<<filename.left(filename.lastIndexOf(".wav"))+"_done.wav";
                rt =file.rename(filename.left(filename.lastIndexOf(".wav"))+"_done.wav");
                if(rt)
                {
                    //qDebug()<<file.fileName()<<endl;
                    emit turnComplete(file.fileName());
                    break;
                }
                count++;
                QEventLoop eventloop;
                QTimer::singleShot(500, &eventloop, SLOT(quit()));
                eventloop.exec();
            }
            cout<<rt<<endl;
            cout<<count<<endl;
            if(!rt)
            {
                cout<<"spyThread Error: file can not be renamed."<<endl;
                return ;
            }

        }
/*
        cout<<" Bytes Filename"<<endl;
        for(int i=0;i<list.size();i++)
        {
            QFileInfo fileInfo = list.at(i);
            std::cout << qPrintable(QString("%1 %2").arg(fileInfo.size(), 10)
                                                    .arg(fileInfo.fileName()));
            std::cout << std::endl;
        }
*/
        QEventLoop eventloop;
        QTimer::singleShot(1000, &eventloop, SLOT(quit())); //wait 1s
        eventloop.exec();

    }

    emit threadstopped("spyThread",false);
    cout<<"spyThread stopped."<<endl;


    return;
}

cSpyThread::cSpyThread(QString wav_seg_path):wav_seg_path(wav_seg_path),is_loop(true)
{

}

int cSpyThread::stop()
{
   // cout<<"this is spyThread speaking : thread stop"<<endl;
    cout<<"stopping spyThread..."<<endl;
    is_loop=false;
    return 0;
}
