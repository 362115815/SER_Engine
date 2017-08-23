QT += core
QT -= gui

CONFIG += c++11

TARGET = voice_emotion_example
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0



unix:!macx: LIBS += -L$$PWD/../../lib/ -lvoice_emotion

INCLUDEPATH += $$PWD/../../include
DEPENDPATH += $$PWD/../../include

unix:!macx: PRE_TARGETDEPS += $$PWD/../../lib/libvoice_emotion.a






#unix:!macx: LIBS += -L$$PWD/../../../build-voice_emotion-Desktop_Qt_5_9_1_GCC_64bit-Release/ -lvoice_emotion

#INCLUDEPATH += $$PWD/../../../Voice_Emotion_Release/include
#unix:!macx: PRE_TARGETDEPS += $$PWD/../../../build-voice_emotion-Desktop_Qt_5_9_1_GCC_64bit-Release/libvoice_emotion.a

unix:!macx: LIBS += -L$$PWD/../../../../anaconda3/lib/ -lpython3.6m
unix:!macx: LIBS += -L$$PWD/../../../../anaconda3/lib/ -licui18n
unix:!macx: LIBS += -L$$PWD/../../../../anaconda3/lib/ -licuuc
unix:!macx: LIBS += -L$$PWD/../../../../anaconda3/lib/ -licudata

INCLUDEPATH += $$PWD/../../../../anaconda3/include/python3.6m


