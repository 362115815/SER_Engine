#ifndef SER_ENGINE_H
#define SER_ENGINE_H
#include"tensorflowcomp.h"
#include <QObject>
#include"opensmilecomp.h"
#include<QString>
#include<QThread>
#include"string"
using std::string;

class cSpyThread;

class cSER_Engine:public QObject
{
    Q_OBJECT
public:
    cSER_Engine(QString work_dir=".");
    ~cSER_Engine();
    int start_engine();
    int stop_engine();
    int set_workdir(QString work_dir);
private:
    QString engine_path;
    QString model_path;
    QString tempdata_path;
    QString wav_seg_path;
    QString feature_path;
    QString openSmile_path;
    QString work_dir;
    cSpyThread* spyThread;
    bool spyThread_running;
    cOpenSmileComp *openSmile;
	cTensorFlowComp * tensorFlow;
    bool openSmile_runnig;
    bool tensorFlow_runnig;
signals:
    void stop_all();
    void recognition_complete(QString wav_path,QString predict_path);
	void engine_started();
public slots:
    int state_recv(QString compName,bool state);

};


class cSpyThread:public QThread
{
    Q_OBJECT

public:
    QString wav_seg_path;
    cSpyThread(QString wav_seg_path);
    void run();

signals:
    void turnComplete(QString wavfile_path);
    void threadstopped(QString name,bool state);
    void threadstarted(QString name,bool state);


public slots:
    int stop();
private:
    bool is_loop;

};

#endif // SER_ENGINE_H
