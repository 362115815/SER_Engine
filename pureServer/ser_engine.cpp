#include "ser_engine.h"
#include <QtQml>
#include"opensmilecomp.h"
#include<iostream>
#include<QEventLoop>
#include<QDir>
#include<QFile>
#include<QLockFile>
#include<QDebug>
#include"clearfolders.h"


using std::cout;
using std::endl;



cSER_Engine::cSER_Engine(QString work_dir)//work_dir目录必须是全路径，且目录下面有SER_Engine文件夹
{
#ifdef OUT_LOG
    fout_recv.open("/home/emo/SER/SER_Server_Release/SER_Engine/log/sample_recv.txt");
    fout_get_one.open("/home/emo/SER/SER_Server_Release/SER_Engine/log/get_one.txt");
    fout_send.open("/home/emo/SER/SER_Server_Release/SER_Engine/log/result_send.txt");
    sample_handled_num=0;
    sample_recv_num=0;
    get_one_num=0;
#endif
    max_recognition_num=128;
    set_workdir(work_dir);
    for(quint64 i=0;i<max_recognition_num;i++)
    {
        cRecSample * recSample=new cRecSample(i);
        connect(recSample,SIGNAL(err_occured(QString,QString,cRecSample&)),this,SLOT(exception_handle(QString,QString,cRecSample&)));
        recognition_pool.append(recSample);
        if(i!=0)
        {
            available_sample_id.enqueue(i);
        }
    }


    spyThread_running=false;
    openSmile_runnig=false;
    tensorFlow_runnig=false;
    engine_running=false;
    engine_exstart=false;

    ckpt_mode=true;
   // cout<<"This is Engine speaking.wav_seg_path=" << wav_seg_path.toStdString()<<endl;
    spyThread=new cSpyThread(work_dir);
    tensorFlow=new cTensorFlowComp(work_dir);
    openSmile=new cOpenSmileComp(work_dir);

    server=new QTcpServer;
   // socket=new QTcpSocket;






    connect(spyThread,SIGNAL(threadstarted(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(spyThread,SIGNAL(threadstopped(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(spyThread,SIGNAL(turnComplete(QString)),openSmile,SLOT(fea_extract(QString)));

    connect(tensorFlow,SIGNAL(recogition_complete(QString)),this,SIGNAL(recognition_complete(QString)));
    connect(tensorFlow,SIGNAL(started(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(tensorFlow,SIGNAL(stopped(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(tensorFlow,SIGNAL(out_predict_result(cRecSample&,QString,QStringList)),this,SIGNAL(out_predict_result(cRecSample&,QString,QStringList)));
    connect(tensorFlow,SIGNAL(out_predict_result(cRecSample&,QString,QStringList)),this,SLOT(send_predict_result(cRecSample&,QString,QStringList)));
    connect(tensorFlow,SIGNAL(err_occured(QString,QString,cRecSample&)),this,SLOT(exception_handle(QString,QString,cRecSample&)));

    connect(openSmile,SIGNAL(err_occured(QString,QString,cRecSample&)),this,SLOT(exception_handle(QString,QString,cRecSample&)));
    connect(openSmile,SIGNAL(recorder_started(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(openSmile,SIGNAL(recorder_stopped(QString,bool)),this,SLOT(state_recv(QString,bool)));
    connect(openSmile,SIGNAL(fea_extra_finished(cRecSample &)),tensorFlow,SLOT(run_trial(cRecSample&)));

    connect(this,SIGNAL(stop_all()),spyThread,SLOT(stop()));
    connect(this,SIGNAL(stop_all()),tensorFlow,SLOT(stop_tf()));
    connect(this,SIGNAL(stop_all()),openSmile,SLOT(stop_recorder()));
    connect(this,SIGNAL(fea_extact(cRecSample &)),openSmile,SLOT(fea_extract(cRecSample &)));
    connect(this,SIGNAL(new_sample_coming(cRecSample*)),openSmile,SLOT(new_sample_coming(cRecSample*)));
    connect(this,SIGNAL(err_occured(QString,QString,cRecSample&)),this,SLOT(exception_handle(QString,QString,cRecSample&)));
     //socket信号绑定

     connect(server,SIGNAL(newConnection()),this,SLOT(handle_new_connection()));



     Py_Initialize();
    //f.setFileName("/home/emo/filerecv.txt");
    //f.open(QFile::WriteOnly);
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
 int cSER_Engine::get_one_to_run()//get a sample to be handled
 {

     if(sample_to_recognize.isEmpty())
     {
         return -1;
     }


     QDateTime time = QDateTime::currentDateTime();//获取系统现在的时间
     QString str = time.toString("yyyy-MM-dd hh:mm:ss ddd"); //设置显示格式


     QString filepath=sample_to_recognize.dequeue();
     get_one_num++;
     fout_get_one<<"get_one_num="<<get_one_num<<"\t"<<filepath.toStdString();


     cout<<str.toStdString()+":"+"SER Server process file: "<<filepath.toStdString()<<endl;

     QString filename= filepath.mid(filepath.lastIndexOf("/")+1);
      filename = filename.left(filename.lastIndexOf(".wav"));
     cout<<"filename="<<filename.toStdString()<<endl;
     cout<<"applying a sample id for new sample"<<endl;

     quint64 id=get_available_id();
     if(id==0)
     {
         cout<<"no id available"<<endl;
         recognition_pool[0]->init(filename,filepath);
         exception_handle("cSER_sever","Server is busy",*recognition_pool[0]);
         QEventLoop eventloop;
         QTimer::singleShot(1000, &eventloop, SLOT(quit())); //wait 1s
         eventloop.exec();
         fout_get_one<<"\t sample state="<<0<<endl;
         return -1;
     }
     cout<<"sample id="<<id<<endl;
     recognition_pool[id]->init(filename,filepath);
     fout_get_one<<"\t sample state="<<recognition_pool[id]->state<<endl;

     emit new_sample_coming(recognition_pool[id]);

     return 0;
 }
  int cSER_Engine::resetSample(quint64 id)
  {
      if(id==0)
      {
          return -1;
      }
      recognition_pool[id]->reset();   
      available_sample_id.enqueue(id);
      return 0;
  }

quint64 cSER_Engine::get_available_id()
{
    quint64 id=0;
    if(!available_sample_id.isEmpty())
    {
        id=available_sample_id.dequeue();
    }

    return id;
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
     if(!ckpt_mode)
     {
         tensorFlow->start();

     }
    else
     {
         tensorFlow->start_ckpt();
     }
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
    connect(socket,SIGNAL(disconnected()),this,SLOT(socket_read_data()));
    //connect(socket,SIGNAL(disconnected()),socket,SLOT(deleteLater()));
    cout<<"new connection handled"<<endl;


    return 0;
}


int cSER_Engine::socket_read_data()//读取socket的数据
{
    QByteArray buffer;
    buffer=socket->readAll();
    QString clientdata(buffer);
   // QTextStream out(&f);
   // out<<clientdata;


    QDateTime time = QDateTime::currentDateTime();//获取系统现在的时间
    QString str = time.toString("yyyy-MM-dd hh:mm:ss ddd"); //设置显示格式
    cout<<endl<<endl<<endl;
    cout<<str.toStdString()+":"+"receiving data from client: "<<clientdata.toStdString()<<endl;
    QStringList filelist=clientdata.split("\n",QString::SkipEmptyParts);
   // qDebug()<<"filelist="<<filelist<<"\tsize="<<filelist.size()<<endl;
    cout<<"filelist:"<<endl;
    for(qint64 i =0;i<filelist.size();i++)
    {
        sample_to_recognize.enqueue(filelist.at(i));
        cout<<filelist.at(i).toStdString()<<endl;
        sample_recv_num++;
        #ifdef OUT_LOG
        fout_recv<<"sample_recv_num="<<sample_recv_num<<"\t"<<filelist.at(i).toStdString()<<endl;
        fout_recv.flush();
        #endif
       // sample_recv_num++;
       // fout1<<"sample_recv_num="<<sample_recv_num<<"\t"<<filelist.at(i).toStdString()<<endl;
        get_one_to_run();
    }



    return 0;
}

int cSER_Engine::exception_handle(QString comp, QString err_msg, cRecSample &sample)//error handle
{
    if(sample.state==0)
    {
        return -1;
    }
    sample_handled_num++;
    cout<<"SER_Server: ERROR : Compnent: "<<comp.toStdString()<<"  Exception: "<<err_msg.toStdString()+"  Filename:"<<sample.filename.toStdString()<<endl;
    cout<<"sample_handled_num="<<sample_handled_num<<"\tsample id="<<sample.id<<":sending error msg to client..."<<endl;
    QString out_data="ERROR:  Compnent:"+comp+"  Exception:"+err_msg+"  Filename:"+sample.filename+"\n";
   // socket->write(out_data.toLatin1());
   // socket->flush();

    fout_send<<"sample_handled_num="<<sample_handled_num<<"\t sample id="<<sample.id<<"\t"<<out_data.toStdString();
    fout_send.flush();
    cout<<"error msg sended."<<endl;
    resetSample(sample.id);
    get_one_to_run();
    return 0;
}

int cSER_Engine::send_predict_result(cRecSample&sample,QString predict,QStringList probability)//发送识别结果
{
    if(sample.state==0)
    {
        return -1;
    }
    sample_handled_num++;
    QString voice_seg=sample.filename;
    cout<<"sample_handled_num="<<sample_handled_num<<"\t sample id="<<sample.id<<":sending predicted result..."<<endl;
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
    fout_send<<"sample_handled_num="<<sample_handled_num<<"\t"<<out_data.toStdString();
    fout_send.flush();
    cout<<"predicted result sent"<<endl;
    resetSample(sample.id);
    get_one_to_run();
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
/*
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
*/
    return 0;
}

 cSER_Engine::~cSER_Engine()
{

    fout_get_one.close();\
    fout_recv.close();
    fout_send.close();

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
    for(quint64 i=0;i<max_recognition_num;i++)
    {
        delete recognition_pool[i];
    }
   // Py_FinalizeEx();
    Py_Finalize();
    f.close();
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
