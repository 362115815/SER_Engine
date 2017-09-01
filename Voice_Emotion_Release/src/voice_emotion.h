#ifndef VOICE_EMOTION_H
#define VOICE_EMOTION_H
#include"tensorflowcomp.h"
#include <QObject>
#include"opensmilecomp.h"
#include<QString>
#include"string"
using std::string;

class cEmoResult
{
public:
   cEmoResult();
   string wav_name;//wav basename
   string emo_label;//ang,hap,neu,sad
   float emo_prob[4];//probabilities of each emotion class, in order of ang, hap, neu, sad
   void reset();
};


class cSER_Engine:public QObject
{
    Q_OBJECT
public:


    /*
    usage: constructor.
    parameters:
        work_dir: path of the directory which contains a folder named SER_Engine.
    */
    cSER_Engine(QString work_dir=".");


    /*
    usage: initiate the engine,including model loading and temporal data clearing.
    parameters:
        work_dir: full path of the directory which contains a folder named SER_Engine.
    return:0 if success, otherwise -1.
    */
    int initiate();


    /*
    usage:get a predicted emotion label of the input wav file.
    parameters:
        emo_result: emotion recognition result
        wav_path: path of the wav file to be recognized, wav format should be in 16kHz, 16bit, mono.
    return: 0 if success, otherwise -1.
    */
    int voice_emo_recog( cEmoResult &  emo_result,const string wav_path);




    virtual ~cSER_Engine();

private:
    bool engine_running;
    bool engine_exstart;
    int set_workdir(QString work_dir=".");

    QString engine_path;
    QString model_path;
    QString tempdata_path;
    QString wav_seg_path;
    QString feature_path;
    QString openSmile_path;
    QString work_dir;
    QString result_dir;
    cOpenSmileComp *openSmile;
    cTensorFlowComp * tensorFlow;
    QProcess * pro;
    bool openSmile_runnig;
    bool tensorFlow_runnig;
private slots:
    int state_recv(QString compName,bool state);


};
#endif // VOICE_EMOTION_H
