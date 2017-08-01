#ifndef DATASOURCE_H
#define DATASOURCE_H

#include <QObject>
#include <QtCharts/QAbstractSeries>
#include <QtCharts/QXYSeries>
QT_CHARTS_USE_NAMESPACE

class cDataSource:public QObject
{
    Q_OBJECT

private :
    QVector<QPointF> * point_list;
    qreal width;
    qreal step;
public:
    quint64 point_num;

    cDataSource(quint64 point_num);
    ~cDataSource();
    void update(QAbstractSeries *series);
    void push(qreal value);
};

#endif // DATASOURCE_H
