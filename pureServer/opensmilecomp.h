#ifndef OPENSMILECOMP_H
#define OPENSMILECOMP_H

#include <QObject>
#include<QProcess>
#include<QString>
#include<QQueue>
#include<QThread>
#include"recsample.h"
#include<QVector>
#include<fstream>
class cFeaExtractThread;

class cOpenSmileComp:public QObject
{
    Q_OBJECT
private:
    quint64 sample_recv_num;
    quint64 sample_handled_num;
    std::ofstream fout_handled;
    std::ofstream fout_recv;
    QProcess * pro;
    QString opensmile_path;
    QString feature_path;
    QString console_msg;
    QQueue<QString> wav_queue;
    QString workdir;
    QString engine_path;
    QString tempdata_path;
    QString wav_seg_path;
    QQueue<cRecSample *> samples_toExtract;//queue of samples  to be handle
    quint64 max_thread_num;
    QQueue<quint64> available_thread_id;// queue that contain the available thread ids.
    QVector<cFeaExtractThread*> FE_thread_pool;//feature extraction thread pool
    quint64 get_available_thread();//return 0,if no free thread to use ,else return thread id;
    int get_one_to_run();//get a sample and run feature extraction,return 0 throw a sample to run, -1 no sample is been thrown
    void resetFEThread(quint64 id);
public:
    void set_workdir(QString workdir);
    cOpenSmileComp(QString workdir=".");
   virtual ~cOpenSmileComp();
    int start_record(QString workdir);//开始录音
    int start_record(int startIndex=1);//开始录音
    int stop_record();//停止录音

public slots:

    int thread_finished(quint64 id);
    void stop_recorder();//停止录音
    int fea_extract(cRecSample & sample);//提取特征
    int fea_extract(QString wavfile);//提取特征
    int new_sample_coming(cRecSample *sample);//handle new sample
signals:
    void recorder_started(QString name,bool state);
    void recorder_stopped(QString name,bool state);
    void fea_extra_finished(cRecSample & sample);//特征提取
    void err_occured(QString Comp,QString msg,cRecSample&);
private slots:
   void readOutput();//读取输出



};

class cFeaExtractThread:public QThread
{
    Q_OBJECT
public:
    quint64 id;
    cFeaExtractThread(quint64 id,QString feature_path,QString opensmile_path);
    ~cFeaExtractThread();
    int state;//thread state,0 free,1 busy
    int fe_state;//0 feature extract succeeded,-1 feature extract failed;
    cRecSample * sample;
    int start_FExtract(cRecSample * sample);//start feature
    QString feature_path;
    QString opensmile_path;
    void reset();
signals:
    void err_occured(QString Comp,QString msg,cRecSample&);
    void thread_finished(quint64 id);
private slots:
    void timer_timeout();
    void process_finished(int exitCode,QProcess::ExitStatus exitStatus);
protected:
    void run();

private:
    QTimer* timer;
    QProcess* pro;
    bool running;

};


#endif // OPENSMILECOMP_H
