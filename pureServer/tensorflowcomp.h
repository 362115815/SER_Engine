#ifndef TENSORFLOWCOMP_H
#define TENSORFLOWCOMP_H
#include<Python.h>
#include <QObject>
#include<QStringList>
#include"recsample.h"
class cTensorFlowComp:public QObject
{
    Q_OBJECT
public:
    cTensorFlowComp(QString work_dir=".");
    virtual ~cTensorFlowComp();
    int start();
    int start_ckpt();
    int stop();
    void set_workdir(QString work_dir);
private:
    QString model_path;
    QString work_dir;
    QString engine_path;
    QString result_dir;
    bool tf_module_imported;
public slots:
    void stop_tf();
    void run_trial(cRecSample & sample);

signals:
    void recogition_complete(QString predict_path);
    void started(QString compName,bool state);
    void stopped(QString compName,bool state);
    void out_predict_result(cRecSample & sample,QString predict,QStringList probability);
    void err_occured(QString Comp,QString msg,cRecSample& sample);
};

#endif // TENSORFLOWCOMP_H
