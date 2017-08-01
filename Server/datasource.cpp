#include "datasource.h"
#include<iostream>

cDataSource::cDataSource(quint64 point_num)
{
    this->point_num=point_num;
    point_list=new QVector<QPointF>;
    QPointF p(0,0);
    for(quint64 i=0;i<point_num;i++)
    {
        p.setX(i);
        point_list->append(p);
    }

}
void cDataSource::push(qreal value)
{
  //  std::cout<<"signal value:"<<value<<std::endl;


    for(quint64 i=0;i<point_num-1;i++)
    {
        (*point_list)[i].setY((*point_list)[i+1].y());
        //std::cout<<"point "<<i<<": x= "<< (*point_list)[i].x()<<",y = "<<(*point_list)[i].y()<<std::endl;

    }
    if(abs(value)>0.05)
    {
         (*point_list)[point_num-1].setY(0);
        return;
    }
    (*point_list)[point_num-1].setY(value);
    std::cout<<"push value:"<<value<<std::endl;

    return ;
}

cDataSource::~cDataSource()
{
    delete point_list;
}


void cDataSource::update(QAbstractSeries *series)
{

    if (series) {
        QXYSeries *xySeries = static_cast<QXYSeries *>(series);
        xySeries->replace(*point_list);
    }
    return;
}

