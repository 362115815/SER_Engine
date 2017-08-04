#ifndef CLEARFOLDERS_H
#define CLEARFOLDERS_H
//头文件包含
#include <QDir>
#include <QFileInfo>
#include <QString>
#include <QList>
#include <QFileInfo>
#include <QFile>
/*删除文件夹内容，不删除该文件夹本身
//入口参数：const QString &folderDir---------------------文件夹全路径
//出口参数：无
//返回值：true----删除成功；false----文件夹不存在
//备注：无*/
bool removeFolderContent(const QString &folderDir);


#endif // OPENSMILECOMP_H
