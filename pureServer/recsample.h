#ifndef RECSAMPLE_H
#define RECSAMPLE_H

#include <QObject>
#include<QTimer>
class cRecSample:public QObject
{
    Q_OBJECT
public:
    explicit cRecSample(quint64 id,QObject *parent = nullptr);
    virtual ~cRecSample();
    void init(QString filename,QString filepath);
    void reset();
    QString filename;
    QString filepath;
    QString feafile;
    QString resultfile;
    quint64 id;
    QTimer* timer;
    int state;
signals:
    void err_occured(QString Comp,QString msg,cRecSample&);
private slots:
    void timer_timeout();
};

#endif // RECSAMPLE_H
