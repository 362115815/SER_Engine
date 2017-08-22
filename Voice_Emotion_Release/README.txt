1. Introduction

   This package provides a library of voice emotion recognition which can be integrated with other projects.

2. Runtime environment

   2.1 The library is developed on Qt platform, the specific version of Qt is 5.9.1. So firstly you should have Qt installed and set the system environment properly.

   2.2 The library also depends on Python and the Tensorflow library because we use Tensorflow to run our recognition model.The specific version of Python is 3.6.0 and 1.2.1 for Tensorflow. We recommend you install Anaconda with Python3.6 and install Tensorflow through Anaconda(Tutorial see www.tensorflow.org)

   2.3 After installing the aforementioned dependencies, add ./engine/SER_Engine/pyscript/tf_module.py to PYTHONPATH to make sure the module tf_module.py can be imported successfully in Python.

IMPORTANT: Please make sure all the binary files in ../engine have executable authority!


3. General usage

   3.1 Firstly, you should add ../include and ../lib to your project including path.

   3.2 Secondly, include ../include/voice_emotion.h in your source file.

   IMPORTANT: Please make sure that voice_emotion.h is the first header file included in your project or it may cause problems during compiling.

   3.3 Declare an instance of the class cSER_Engine and call the member function initiate() to do some initiation work(loading model,clearing temporal data...)

   3.4 Use the member function voice_emo_recog() of cSER_Engine to get a predicted emotion label of the input wav file.

4. Example see ../example/voice_emotion_example
