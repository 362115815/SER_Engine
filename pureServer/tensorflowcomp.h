#ifndef TENSORFLOWCOMP_H
#define TENSORFLOWCOMP_H
#include<Python.h>
#include <QObject>
#include<QStringList>

class cTensorFlowComp:public QObject
{
    Q_OBJECT
public:
    cTensorFlowComp(QString work_dir=".");
    virtual ~cTensorFlowComp();
    int start();
    int stop();
    void set_workdir(QString work_dir);
private:
    QString model_path;
    QString work_dir;
    QString engine_path;
    QString result_path;
    bool tf_module_imported;
public slots:
    void stop_tf();
    void run_trial(QString fea_path);

signals:
    void recogition_complete(QString predict_path);
    void started(QString compName,bool state);
    void stopped(QString compName,bool state);
    void out_predict_result(QString voice_seg,QString predict,QStringList probability);
    void err_occured(QString Comp,QString msg);
};

#endif // TENSORFLOWCOMP_H
