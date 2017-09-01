#-------------------------------------------------
#
# Project created by QtCreator 2017-08-21T16:44:00
#
#-------------------------------------------------

QT       -= gui

TARGET = voice_emotion
TEMPLATE = lib
CONFIG += staticlib

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        voice_emotion.cpp \
    clearfolders.cpp \
    opensmilecomp.cpp \
    tensorflowcomp.cpp

HEADERS += \
        voice_emotion.h \
    clearfolders.h \
    opensmilecomp.h \
    tensorflowcomp.h
unix {
    target.path = /usr/lib
    INSTALLS += target
}

unix:!macx: LIBS += -L$$PWD/../../anaconda3/lib/ -lpython3.6m

INCLUDEPATH += $$PWD/../../anaconda3/include/python3.6m

unix:!macx: LIBS += -L$$PWD/../../anaconda3/lib/ -licui18n
unix:!macx: LIBS += -L$$PWD/../../anaconda3/lib/ -licuuc
unix:!macx: LIBS += -L$$PWD/../../anaconda3/lib/ -licudata
