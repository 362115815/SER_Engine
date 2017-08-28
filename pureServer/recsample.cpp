#include "recsample.h"

cRecSample::cRecSample(quint64 id, QObject *parent): QObject(parent)
{
    filename="";
    filepath="";
    feafile="";
    resultfile="";
    state=0;
    this->id=id;
    timer=new QTimer;
    timer->setInterval(5000);
    timer->setSingleShot(true);
    connect(timer,SIGNAL(timeout()),this,SLOT(timer_timeout()));
    return ;

}
cRecSample::~cRecSample()
{
   delete timer;
   return;
}

void cRecSample::timer_timeout()
{
    emit err_occured("cRecSample","Timer time out",*this);
}

void cRecSample::init(QString filename,QString filepath)
{
    this->filename=filename;
    this->filepath=filepath;
    this->state=1;
    this->feafile="";
    this->resultfile="";
    timer->start();
    return;
}

void cRecSample::reset()
{
    filename="";
    filepath="";
    resultfile="";
    feafile="";
    state=0;
    timer->stop();
    return;
}

