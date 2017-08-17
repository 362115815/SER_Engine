#include <QCoreApplication>
#include <QProcess>
#include <QThread>
#include <QDebug>
#include <QFile>
#include <QTimer>
#include <QMutex>


#define  ProcessNum  24                               //process number

QMutex mutex;
/*
QStringList argu;                                      // opensmile parameter
QString inputfile;                                     // inputfilename
QString outputfile;                                    // outputfile name
*/
QStringList fonts;
int n = ProcessNum-1;
QProcess *proc[ProcessNum];
QFile filelist("/data/mm0105.chen/wjhan/database/record_openInn_NOISE.list");                       // inputfile list
QString confile = "/data/mm0105.chen/wjhan/xiaomin/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf";            // opensmile configuration's file
QString program = "/data/mm0105.chen/wjhan/xiaomin/opensmile/bin/SMILExtract";          // opensmile exe
QString in_path="/data/mm0105.chen/wjhan/database/record_openInn_NOISE/";
QString out_path="/data/mm0105.chen/wjhan/database/record_openInn_NOISE_feature/";

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    if(filelist.open(QFile::ReadOnly | QIODevice::Text)) {
        QTextStream data(&filelist);
        QString line;
        while (!data.atEnd()){
            line = data.readLine();
            line.remove('\n');
            fonts<<line;
        }
   }
      QStringList argu;
    for( int i = 0 ; i < ProcessNum; i++){

       argu.clear();
       QString inputfile = in_path+fonts[i];

       QString outputfile = QString(out_path+"%1.csv").arg(fonts[i]).remove(".wav");

      QString instname=QString(fonts[i]).remove(".wav");
      QString  classlabel=fonts[i].section("_",3,3).left(3);

      // qDebug()<<"instname="<<instname;
       //qDebug()<<"classlabel="<<classlabel;

      argu<<"-C"<<confile<<"-I"<<inputfile<<"-O"<<outputfile<<"-classes"<<"{ang,nor,hap,sad}"<<"-classlabel"<<classlabel<<"-instname"<<instname;

       proc[i] = new QProcess();
       qDebug()<<"i:"<<i<<argu;
      QObject:: connect(proc[i], static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), [i] (int exitCode, QProcess::ExitStatus exitStatus){
            QStringList argu;
           if(n < (fonts.size()-1))
           {
                mutex.lock();
                n++;

                qDebug()<<"n"<<n;
                qDebug()<<"i"<<i;
                qDebug()<<"fonts"<<fonts.size();
                qDebug()<<"exitCode"<<exitCode;
                qDebug()<<"Exit status: " << exitStatus;
                qWarning() << QString("proc[%1] restart... ").arg(i);
              QString  inputfile = in_path+fonts[n];
              QString  instname=QString(fonts[n].remove(".wav"));
              QString  classlabel=fonts[n].section("_",3,3).left(3);
              QString  outputfile = QString(out_path+"%1.csv").arg(fonts[n]).remove(".wav");

              qDebug()<<"n:"<<n<<argu;
              argu<<"-C"<<confile<<"-I"<<inputfile<<"-O"<<outputfile<<"-classes"<<"{ang,nor,hap,sad}"<<"-classlabel"<<classlabel<<"-instname"<<instname;
              mutex.unlock();
                proc[i]->start(program, argu);
           }else{
               if(proc[i]) {
                   proc[i]->close();
                   delete proc[i];
               }
               qDebug()<<QString("proc[%1] Done!!!!!!!!!!!!!!!!!!!!!!!!!").arg(i);
           }
       });

       proc[i]->start(program, argu);

    }




    return a.exec();
}
