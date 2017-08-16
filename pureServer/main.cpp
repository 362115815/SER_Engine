#include "ser_engine.h"
#include <QCoreApplication>
//#include <QGuiApplication>
//#include <QQmlApplicationEngine>

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    //QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    //QGuiApplication app(argc, argv);
   // qmlRegisterType<cSER_Engine>("SERComp", 1, 0, "SER_Engine");
   // QQmlApplicationEngine engine;
   // engine.load(QUrl(QLatin1String("qrc:/main.qml")));
  //  if (engine.rootObjects().isEmpty())
   //     return -1;

   QString work_dir=argv[1];
   qint64 port=atoi(argv[2]);
   cSER_Engine ser_engine(work_dir);
    setvbuf(stdout, (char *)NULL, _IONBF, 0);
    setvbuf(stderr, (char *)NULL, _IONBF, 0);
   int rt=ser_engine.start_asServer(work_dir,port);
    if(rt!=0)
    {
        exit(rt);
    }
    return app.exec();
}
