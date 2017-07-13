#ifndef OPENSMILECOMP_H
#define OPENSMILECOMP_H

#include <QObject>
#include<QProcess>
#include<QString>
#include<QQueue>
class cOpenSmileComp:public QObject
{
    Q_OBJECT
private:
    QProcess * pro;
    QString opensmile_path;
    QString feature_path;
    QString console_msg;
    QQueue<QString> wav_queue;
    QString workdir;
    QString engine_path;
    QString tempdata_path;
    QString wav_seg_path;

    void setWorkdir(QString workdir);
public:
    cOpenSmileComp(QString workdir="./");
    ~cOpenSmileComp();
    int start_record(QString workdir = "./");//开始录音
    int stop_record();//停止录音
public slots:
    void stop_recorder();//停止录音
    int fea_extract(QString wavfile);//提取特征
signals:
    void recorder_started(QString name,bool state);
    void recorder_stopped(QString name,bool state);
    void fea_extra_finished(QString wav_path, QString fea_path);//特征提取完毕
private slots:
   void readOutput();//读取输出



};

#endif // OPENSMILECOMP_H
