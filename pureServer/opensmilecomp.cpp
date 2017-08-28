#include "opensmilecomp.h"
#include<iostream>
#include<string>
#include<cstdio>
#include<QEventLoop>
#include<QTimer>
#include<QDebug>
#include<QFile>

using std::cout;
using std::endl;
using std::string;
//extern int log_level;
cOpenSmileComp::cOpenSmileComp(QString workdir)
{
    set_workdir(workdir);

    sample_recv_num=0;

    fout_recv.open("/home/emo/SER/SER_Server_Release/SER_Engine/log/opensmile_recv.txt");
    fout_handled.open("/home/emo/SER/SER_Server_Release/SER_Engine/log/opensmile_handled.txt");
    max_thread_num=8;

    /*
    cout<<"work_dir="<<workdir.toStdString()<<endl;
    cout<<"feature_path="<<feature_path.toStdString()<<endl;
    cout<<"opensmile_path="<<opensmile_path.toStdString()<<endl;
    exit(1);
    */
    for(quint64 i=0;i<max_thread_num;i++)
    {
        cFeaExtractThread * thread=new cFeaExtractThread(i,feature_path,opensmile_path);
        connect(thread,SIGNAL(err_occured(QString,QString,cRecSample&)),this,SIGNAL(err_occured(QString,QString,cRecSample&)));
        connect(thread,SIGNAL(thread_finished(quint64)),this,SLOT(thread_finished(quint64)));
        FE_thread_pool.append(thread);
        if(i!=0)
        {
            available_thread_id.enqueue(i);
        }
    }


    pro=NULL;
    console_msg="";
    return;
}

 void cOpenSmileComp::set_workdir(QString workdir)
 {
     //cout<<"opensmile setting workdir:"<<workdir.toStdString()<<endl;
     //cout<<&(this->workdir)<<endl;
     this->workdir=workdir;
     engine_path=workdir+"/SER_Engine";
     opensmile_path=engine_path+"/opensmile";
     tempdata_path=engine_path+"/tempdata";
     wav_seg_path=tempdata_path+"/wav_seg";
     feature_path=tempdata_path+"/feature";
    // cout<<"opensmile set up workdir:"<<workdir.toStdString()<<endl;
   /*  QString opensmile_path;
     QString feature_path;
     QString console_msg;
     QQueue<QString> wav_queue;
     QString workdir;
     QString engine_path;
     QString tempdata_path;
     QString wav_seg_path;
     */



     return;
 }
int cOpenSmileComp::start_record(int startIndex)
{
    cout<<"startIndex="<<startIndex<<endl;

    QString cmd=this->opensmile_path+"/bin/SMILExtractPA -C "+opensmile_path+"/config/myconfig/live_rec.conf"+\
            " -filebase "+wav_seg_path+"/output_segment_"+" -startindex "+QString::number(startIndex,10);

    cout<<"record cmd:"<<cmd.toStdString()<<endl;
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
quint64 cOpenSmileComp::get_available_thread()
{
    quint64 id=0;
    if(!available_thread_id.isEmpty())
    {
        id=available_thread_id.dequeue();
    }
    return id;
}

int cOpenSmileComp::start_record(QString workdir)
{
    set_workdir(workdir);
   // cout<<"thisi opensimile speaking: opensmilepath="<<opensmile_path.toStdString()<<endl;

    QString cmd=this->opensmile_path+"/bin/SMILExtractPA -C "+opensmile_path+"/config/myconfig/live_rec.conf"+\
            " -filebase "+wav_seg_path+"/output_segment_";


    pro =new QProcess(this);
    connect(pro,SIGNAL(readyReadStandardError()),this,SLOT(readOutput()));
    connect(pro,SIGNAL(readyRead()),this,SLOT(readOutput()));
    setvbuf(stdout, (char *)NULL, _IONBF, 0);//关闭缓冲
    pro->setProcessChannelMode(QProcess::MergedChannels);
    std::cout<<cmd.toStdString()<<endl;
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
    if(pro!=NULL)
    {
        cout<<"stopping recorder..."<<endl;
        pro->close();
        string msg=pro->readAll().toStdString();
        cout<<msg<<endl;
        cout<<"recorder stopped."<<endl;
        emit recorder_stopped("openSmile",false);
    }

    return 0;
}

cOpenSmileComp::~cOpenSmileComp()
{
    fout_recv.close();
    fout_handled.close();
    cout<<"!OPENSMILE"<<endl;
    if(pro)
    {
        pro->close();
        delete pro;
    }
    for(quint64 i=0;i<max_thread_num;i++)
    {
        delete FE_thread_pool[i];
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
 void cOpenSmileComp::resetFEThread(quint64 id)
 {
     FE_thread_pool[id]->reset();
     available_thread_id.enqueue(id);
     return ;
 }

 int cOpenSmileComp::get_one_to_run()//get a sample and run feature extraction
 {
     if(samples_toExtract.isEmpty())
     {
         return -1;
     }

     quint64 id=get_available_thread();
     if(id==0)
     {
        // emit err_occured("openSmileComp","opensmile is busy",*recSample);
         cout<<"opensmile is busy.no free thread to use."<<endl;
         return -1;
     }
     cRecSample * recSample=samples_toExtract.dequeue();
     cout<<"opensmile thread run: id="<<id<<endl;
     FE_thread_pool[id]->start_FExtract(recSample);
     return 0;
 }

 int cOpenSmileComp::new_sample_coming(cRecSample * sample)
 {
      cout<<"This is opensmile speaking : new sample coming :"<<sample->filepath.toStdString()<<endl;
      fout_recv<<"sample_recv_num="<<++sample_recv_num<<"\t"<<sample->filepath.toStdString()<<endl;
      fout_recv.flush();
     samples_toExtract.enqueue(sample);
     get_one_to_run();
     return 0;
 }

int cOpenSmileComp::fea_extract(QString wavfile)//提取特征
{
    Q_UNUSED(wavfile);
    return 0;
}
int cOpenSmileComp::thread_finished(quint64 id)//a thread finished, reset it and add it to the queue of available ids
{

    fout_handled.flush();
    if(FE_thread_pool[id]->sample->state==2)
    {
        fout_handled<<"sample_handled_num="<<++sample_handled_num<<"\t id="<<id<<"\t sample state="<<FE_thread_pool[id]->sample->state \
                   <<"\t"<<FE_thread_pool[id]->sample->filepath.toStdString()<<endl;
        emit fea_extra_finished(*(FE_thread_pool[id]->sample));
    }
    //cout<<"cOpenSmileComp::thread_finished: sample state="<<FE_thread_pool[id]->sample->state<<endl;

    resetFEThread(id);
    get_one_to_run();
    return 0;
}

int cOpenSmileComp::fea_extract(cRecSample & sample)//提取特征
{
    QString wavfile=sample.filepath;
    cout<<"This opensmilecomp speaking: wavfile="<<wavfile.toStdString()<<endl;
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
        //filename = feature_path+filename.left(filename.lastIndexOf("_done.wav")).append(".csv");
        filename = feature_path+filename.left(filename.lastIndexOf(".wav")).append(".csv");
        //qDebug()<<"feaname:"<<feaname;
        QFile file;
        file.setFileName(filename);

        QString cmd=opensmile_path+"/bin/SMILExtractPA -C "+opensmile_path+"/config/gemaps/eGeMAPSv01a.conf"+
                " -I "+wavname+" -csvoutput "+filename;
        qDebug()<<"feature extract cmd:"<<cmd<<endl;
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
                sample.state=2;
                sample.feafile=file.fileName();
                emit fea_extra_finished(sample);
                break;
            }
            count++;
            QEventLoop eventloop;
            QTimer::singleShot(50, &eventloop, SLOT(quit()));
            eventloop.exec();
        }

        //cout<<rt<<endl;
        cout<<count<<endl;
        if(!rt)
        {
            pro.close();
            emit err_occured("OpenSmileComp","file can not be renamed",sample);
            cout<<"fea_extract Error: file can not be renamed."<<endl;
            return -1;

        }

    }



    return 0;
}

int cFeaExtractThread::start_FExtract(cRecSample * sample)//start feature extraction
{
    cout<<"start_FExtract"<<endl;
    this->sample=sample;
    state=1;
    running=true;
    run();
    return 0;
}
void cFeaExtractThread::reset()
{
    this->id=id;
    this->feature_path=feature_path;
    this->opensmile_path=opensmile_path;
    state=0;
    fe_state=-1;
    running=false;
    return;
}

void cFeaExtractThread::run()
{
    QString wavfile=sample->filepath;
    cout<<"this is fea_extarct speaking:";
    cout<<"wavefile:"<<wavfile.toStdString()<<endl;
    QString feafile=feature_path+"/"+sample->filename+".csv";
    sample->feafile=feafile;
    QString cmd=opensmile_path+"/bin/SMILExtractPA -C "+opensmile_path+"/config/gemaps/eGeMAPSv01a.conf"+
            " -I "+wavfile+" -csvoutput "+feafile;
    cout<<"feature extarct cmd:"<<cmd.toStdString()<<endl;
    cout<<"cFeaExtracTread "<<id<<":timer started"<<endl;
    //sleep(2);
    timer->start();
    pro->start(cmd);
    while(running)
    {
        QEventLoop eventloop;
        QTimer::singleShot(50, &eventloop, SLOT(quit()));
        eventloop.exec();
    }



    return;
}
cFeaExtractThread::~cFeaExtractThread()
{

    delete timer;
    delete pro;
}
void cFeaExtractThread::process_finished(int exitCode,QProcess::ExitStatus exitStatus)
{
    running=false;
    Q_UNUSED(exitCode);
    cout<<"cFeaExtractThread "<<id<<":process finished.exitStatus:"<<exitStatus<<endl;
    timer->stop();
    cout<<"timer stopped"<<endl;
    if(exitStatus==QProcess::NormalExit)
    {
       sample->state=2;//feature extraction finished;
       fe_state=0;
    }
    else
    {
        emit err_occured("cFeaExtractThread","SMILExtract crashed",*sample);
    }
    emit thread_finished(id);
    return;
}

void cFeaExtractThread::timer_timeout()
{
    running=false;
    pro->close();
    cout<<"cFeaExtracThread "<<id<<":"<<"timer time out"<<endl;

   // if(QFile::exists(sample->feafile))
    {
      //  QFile::remove(sample->feafile);
    }

    emit err_occured("cFeaExtracThread","timer time out",*sample);
    emit thread_finished(id);
    return;
}

cFeaExtractThread::cFeaExtractThread(quint64 id, QString feature_path,QString opensmile_path)
{
    this->id=id;
    this->feature_path=feature_path;
    this->opensmile_path=opensmile_path;
    state=0;
    fe_state=-1;
    running=false;
    timer=new QTimer;
    timer->setInterval(5000);
    timer->setSingleShot(true);
    pro=new QProcess(this);
    connect(timer,SIGNAL(timeout()),this,SLOT(timer_timeout()));
    connect(pro,SIGNAL(finished(int,QProcess::ExitStatus)),this,SLOT(process_finished(int,QProcess::ExitStatus)));

    return;
}
