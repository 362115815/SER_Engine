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


public:
    void set_workdir(QString workdir);
    cOpenSmileComp(QString workdir=".");
   virtual ~cOpenSmileComp();
    int start_record(QString workdir);
    int start_record(int startIndex=1);
    int start_record_withRTDataOut(int startIndex,QString host_addr,quint64 host_port);
    int stop_record();
public slots:
    void stop_recorder();
    int fea_extract(QString wavfile);
signals:
    void recorder_started(QString name,bool state);
    void recorder_stopped(QString name,bool state);
    void fea_extra_finished(QString fea_path);
private slots:
   void readOutput();



};

#endif // OPENSMILECOMP_H
