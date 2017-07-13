#ifndef TENSORFLOWCOMP_H
#define TENSORFLOWCOMP_H
#include<Python.h>
#include <QObject>

class cTensorFlowComp:public QObject
{
    Q_OBJECT
public:
    cTensorFlowComp(QString work_dir=".");
    int start();
    int stop();
    void set_workdir(QString work_dir);
private:
    QString model_path;
    QString work_dir;
    QString engine_path;
    QString result_path;
public slots:
    void stop_tf();
    void run_trial(QString wav_path,QString fea_path);

signals:
    void recogition_complete(QString wavpath,QString predict_path);
    void started(QString compName,bool state);
    void stopped(QString compName,bool state);
};

#endif // TENSORFLOWCOMP_H
