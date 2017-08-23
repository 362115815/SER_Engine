#include<voice_emotion.h>
#include<iostream>
using std::cout;
using std::endl;
int main(int argc, char *argv[])
{
    cSER_Engine ser_engine("../../engine");
    cEmoResult emo_result;
    int rt=ser_engine.initiate();//initiate the engine
    if(rt!=0)
    {
        return -1;
    }

    ser_engine.voice_emo_recog(emo_result,"../voice_emotion_example/018_sadness_utterance_2022.wav");//run a trial

    cout<<"wave name:"<<emo_result.wav_name<<endl;
    cout<<"emotion label:"<<emo_result.emo_label<<endl;
    cout<<"probability for each emotion:"<<endl;
    for(int i=0;i<4;i++)
    {
        cout<<emo_result.emo_prob[i]<<" ";
    }
    cout<<endl;
    return -1;

}
