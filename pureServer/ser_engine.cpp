#include "ser_engine.h"
#include <QtQml>
#include"opensmilecomp.h"
#include<iostream>
#include<QEventLoop>
#include<QTimer>
#include<QDir>
#include<QFile>
#include<QLockFile>
#include<QDebug>
#include"clearfolders.h"

using std::cout;
using std::endl;

cSER_Engine::cSER_Engine(QString work_dir)//work_dir目录必须是全路径，且目录下面有SER_Engine文件夹
{

    spyThread_running=false;
    openSmile_runnig=false;
    tensorFlow_runnig=false;
    engine_running=false;
    engine_exstart=false;

   // cout<<"This is Engine speaking.wav_seg_path=" << wav_seg_path.toStdString()<<endl;
    spyThread=new cSpyThread(wav_seg_path);
    tensorFlow=new cTensorFlowComp(work_dir);
    openSmile=new cOpenSmileComp(work_dir);
    server=new QTcpServer;
   // socket=new QTcpSocket;

    set_workdir(work_dir);



    connect(spyThread,SIGNAL(threadstarted(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(spyThread,SIGNAL(threadstopped(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(spyThread,SIGNAL(turnComplete(QString)),openSmile,SLOT(fea_extract(QString)));

    connect(tensorFlow,SIGNAL(recogition_complete(QString)),this,SIGNAL(recognition_complete(QString)));
    connect(tensorFlow,SIGNAL(started(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(tensorFlow,SIGNAL(stopped(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(tensorFlow,SIGNAL(out_predict_result(QString,QString,QStringList)),this,SIGNAL(out_predict_result(QString,QString,QStringList)));
    connect(tensorFlow,SIGNAL(out_predict_result(QString,QString,QStringList)),this,SLOT(send_predict_result(QString,QString,QStringList)));
    connect(tensorFlow,SIGNAL(err_occured(QString,QString)),this,SLOT(exception_handle(QString,QString)));

    connect(openSmile,SIGNAL(err_occured(QString,QString)),this,SLOT(exception_handle(QString,QString)));
    connect(openSmile,SIGNAL(recorder_started(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(openSmile,SIGNAL(recorder_stopped(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(openSmile,SIGNAL(fea_extra_finished(QString)),tensorFlow,SLOT(run_trial(QString)));

    connect(this,SIGNAL(stop_all()),spyThread,SLOT(stop()));
    connect(this,SIGNAL(stop_all()),tensorFlow,SLOT(stop_tf()));
    connect(this,SIGNAL(stop_all()),openSmile,SLOT(stop_recorder()));
    connect(this,SIGNAL(fea_extact(QString)),openSmile,SLOT(fea_extract(QString)));

     //socket信号绑定

     connect(server,SIGNAL(newConnection()),this,SLOT(handle_new_connection()));


     Py_Initialize();


}

int cSER_Engine::preboot_engine()
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
    //开启监视线程


    cout<<"starting spyThread..."<<endl;
    //spyThread->set_wav_seg_path(wav_seg_path);

    spyThread->start();
    count=0;
     while(count<5 && !spyThread_running)
       {
           QEventLoop eventloop;
           QTimer::singleShot(2000, &eventloop, SLOT(quit())); //wait 2s
           eventloop.exec();
       }

     if(!spyThread_running)
     {
         cout<<"spyThread start failed."<<endl;
         cout<<"SER Engine pre-boot failed"<<endl;
         emit stop_all();
         return -1;
     }


    //开启 tensorflow module
    cout<<"starting tensorflow"<<endl;

    count=0;
    tensorFlow->start();

    while(count<5 && !tensorFlow_runnig)
    {
        QEventLoop eventloop;
        QTimer::singleShot(2000, &eventloop, SLOT(quit())); //wait 2s
        eventloop.exec();
    }

    if(!tensorFlow_runnig)
    {
        cout<<"tensorFlow start failed."<<endl;
        cout<<"SER Engine pre-boot failed"<<endl;
        emit stop_all();
        return -1;
    }



   cout<<"SER Engine is in exstart state."<<endl;
   engine_exstart=true;
    //QMessageBox::about(NULL, "About", "SER Engine started.");

    return 0;
}

int cSER_Engine::start_asServer(QString work_dir,qint64 port)//作为服务端启动
{
    set_workdir(work_dir);

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



    //开启 tensorflow
    cout<<"starting tensorflow"<<endl;
     count=0;
    tensorFlow->start();

    while(count<3 && !tensorFlow_runnig)
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
        emit stop_all();
        return -1;
    }


    //监听指定的端口
     cout<<"SER Server:binding address..."<<endl;
    if(!server->listen(QHostAddress::LocalHost,port))
    {

        //若出错，则输出错误信息
        cout<<server->errorString().toStdString()<<endl;

        return -1;
    }
    cout<<"SER Server:address bound"<<endl;
    cout<<"SER Engine is listening,port:"<<port<<endl;

    return 0;
}

int cSER_Engine::get_startIndex()//获取录音开始编号
{
    return spyThread->filecur_num+1;
}
 QString cSER_Engine::get_wav_seg_path()
 {
     return wav_seg_path;
 }

int cSER_Engine::stop_recorder()//停止opensmile录音
{
    openSmile->stop_record();
    return 0;
}

int cSER_Engine::start_recorder()
{
    if(!engine_exstart)
    {
        cout<<"recorder start failed. SER_Engine hasn't been pre-boot."<<endl;
        return -1;
    }
/*
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
*/


    //开启 opensmile 录音进程
    cout<<"starting recorder..."<<endl;

    openSmile->start_record(get_startIndex());//启动 opensmile recorder

   int count=0;

    while(count<5 && !openSmile_runnig)
    {
        QEventLoop eventloop;
        QTimer::singleShot(2000, &eventloop, SLOT(quit())); //wait 2s
        eventloop.exec();
        count++;
    }


    if(!openSmile_runnig)
    {
        return -1;
    }






    return 0;
}

int cSER_Engine::start_engine()
{
   if(engine_running)
   {
       cout<<"SER_Engine start failed.Engine is already running."<<endl;
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



    cout<<"\nstrating SER Engine..."<<endl;
    //开启监视线程
    count=0;
    cout<<"starting spyThread..."<<endl;
    //spyThread->set_wav_seg_path(wav_seg_path);
    spyThread->start();
    count=0;

    while(count<5 && !spyThread_running)
    {
        QEventLoop eventloop;
        QTimer::singleShot(2000, &eventloop, SLOT(quit())); //wait 2s
        eventloop.exec();
    }

    //开启 tensorflow module
    tensorFlow->start();
    //开启 opensmile 进程
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



    if(spyThread_running && openSmile_runnig && tensorFlow_runnig)
    {

        cout<<"SER Engine started."<<endl;
        engine_running=true;
        //QMessageBox::about(NULL, "About", "SER Engine started.");
    }
    else
    {
        cout<<"SER Engine start failed"<<endl;
        emit stop_all();
        return -1;
    }

    return 0;
}

int cSER_Engine:: restart_engien()
{
    int i;
    cout<<"restarting SER_Engine..."<<endl;
    if(engine_running)
    {
        i=stop_engine();
        if(i!=0)
        {
            cout<<"SER_Engine restrat failed."<<endl;
            return -1;
        }

    }

     i=start_engine();
     if(i!=0)
     {
         cout<<"SER_Engine restrat failed."<<endl;
         return -1;
     }
     cout<<"SER_Engin restarted."<<endl;
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
        engine_running=true;
        //QMessageBox::about(NULL, "About", "SER Engine stopped.");
    }
    else
    {
        cout<<"SER Engine stop failed"<<endl;
        return -1;
    }

    engine_running=false;

   return 0;
}

int cSER_Engine::handle_new_connection()//处理新的连接
{
/*
    cout<<"handling new connection..."<<endl;
    //release old connection
    if(socket)
    {
        if(socket)
        {
            if(socket->isOpen())
            {
                socket->close();
            }
            delete socket;
        }

    }

*/
    socket=server->nextPendingConnection();

    connect(socket,SIGNAL(readyRead()),this,SLOT(socket_read_data()));
    //connect(socket,SIGNAL(disconnected()),socket,SLOT(deleteLater()));
    cout<<"new connection handled"<<endl;


    return 0;
}


int cSER_Engine::socket_read_data()//读取socket的数据
{

    QByteArray buffer;
    buffer=socket->readAll();
    qDebug()<<"receiving data for client: "<<buffer<<endl;
    //socket->write("data received.");
   // socket->flush();

    emit fea_extact(buffer);
    return 0;
}

int cSER_Engine::exception_handle(QString comp,QString err_msg)//error handle
{
    cout<<"SER_Server: ERROR : Compnent: "<<comp.toStdString()<<"\t Exception: "<<err_msg.toStdString()<<endl;
    cout<<"sending error msg to client..."<<endl;
    QString out_data="ERROR:  Compnent: "+comp+"\t Exception: "+err_msg+"\n";

    socket->write(out_data.toLatin1());
    socket->flush();
    cout<<"error msg sended."<<endl;
    return 1;
}

int cSER_Engine::send_predict_result(QString voice_seg,QString predict,QStringList probability)//发送识别结果
{
    cout<<"sending predict result..."<<endl;
    QString out_data="";

    QString str="";
    for(int i=0;i<probability.size()-1;i++)
    {
        str+=probability[i]+" ";

    }
    str+=probability[probability.size()-1];

    out_data=voice_seg+" "+predict+" "+str+"\n";

    socket->write(out_data.toLatin1());
    socket->flush();
    cout<<"predict result sended"<<endl;

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
    engine_path=work_dir+"/SER_Engine";
    model_path=engine_path+"/model";
    tempdata_path=engine_path+"/tempdata";
    wav_seg_path=tempdata_path+"/wav_seg";
    feature_path=tempdata_path+"/feature";
    openSmile_path=engine_path+"/opensmile";

    if(tensorFlow!=NULL)
    {
        tensorFlow->set_workdir(work_dir);
    }
    if(openSmile!=NULL)
    {
       // cout<<"work_dir="<<work_dir.toStdString()<<endl;
        openSmile->set_workdir(work_dir);
    }
    if(spyThread!=NULL)
    {
        spyThread->set_workdir(work_dir);
    }

    return 0;
}

 cSER_Engine::~cSER_Engine()
{



    if(spyThread_running||openSmile_runnig||tensorFlow_runnig)
    {
        cout<<"Engine destroyed while engine is still running."<<endl;
    }
    if(server)
    {
        if(server->isListening())
           {
            server->close();
        }
        delete server;
    }
    if(socket)
    {
        if(socket->isOpen())
        {
            socket->close();
        }
        delete socket;
    }
    delete openSmile;
    delete tensorFlow;
    delete spyThread;
    //Py_FinalizeEx();
    Py_Finalize();
}

void cSpyThread::set_workdir(QString work_dir)
{
    this->wav_seg_path=work_dir+"/SER_Engine/tempdata/wav_seg";
    return;
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
    is_loop=true;

    while(is_loop)
    {
     // cout<<"This is spyThread speaking :spyThread is running."<<endl;
      // cout<<"wav_seg_path="<<wav_seg_path.toStdString()<<endl;

     //   sleep(3);
        dir.refresh();
        QFileInfoList list=dir.entryInfoList();
         filecur_num=list.size();
        while(filecount<filecur_num)
        {
          //  cout<<"filecur_num:"<<filecur_num<<endl;
           // cout<<"filecount:"<<filecount<<endl;
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
           // cout<<rt<<endl;
            //cout<<count<<endl;
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
        QTimer::singleShot(500, &eventloop, SLOT(quit())); //wait 1s
        eventloop.exec();

    }

    emit threadstopped("spyThread",false);
    cout<<"spyThread stopped."<<endl;


    return;
}
cSpyThread::~cSpyThread()
{
    cout<<"!spyThread"<<endl;
    is_loop=false;
    exit(0); //退出线程
    wait(500);  //等待线程退出，最多500ms

    return;
}


cSpyThread::cSpyThread(QString work_dir):wav_seg_path(work_dir+"/SER_Engine/tempdata/wav_seg")
{
    is_loop=true;
    filecur_num=0;
}

int cSpyThread::stop()
{
   // cout<<"this is spyThread speaking : thread stop"<<endl;
    cout<<"stopping spyThread..."<<endl;
    is_loop=false;
    return 0;
}
