1. SER_Server.class 提供了调用SER(Speech Emotion Recognition) Engine的接口

2. SER_Server 的成员函数：

   2.1 public int start_engine(String work_dir,int port)throws IOException

   简介：这个函数功能是调用 ./bin/SER_Server ，./bin/SER_Server 是SER Engine的服务器版可执行文件。在启动时./bin/SER_Server 先初始化（设置工作目录，载入模型等等),然后监听本地端口，等待识别任务请求。

   参数:
   work_dir: SER_Engine 文件夹所在的路径，如果 path_prefix/SER_Server_Release
   port: ./bin/SER_Server 要监听的本地端口

   返回值：
   成功返回0，失败则抛出异常或非零值。

   注意：运行这个函数后，需要调用public int stop_engine()throws IOException 函数关闭./bin/SER_Server,不然./bin/SER_Server 会一直占用端口进行监听


   2.2 public int post_request(String wav_path)throws IOException

   简介：这个函数向 ./bin/SER_Server 发送识别请求， ./bin/SER_Server 收到请求后进行情感识别，并返回识别结果

   参数：
   wav_path: 待识别的音频文件全路径
   返回值：
   成功返回0，失败则抛出异常或非零值。

   2.3 public String get_result()throws IOException

   简介：这个函数向 ./bin/SER_Server 获取识别结果
   返回值:
   成功返回识别结果，失败返回null

   返回值解析：wave_name emotion_label each_emotion_probability. 返回值是一个用空格隔开的字符串，第一值为音频文件名称，第二个值为识别出来的情感类别，剩下的值为每一类情感对应的概率。
   2.4 public int stop_engine()throws IOException 
   简介：这个函数关闭 ./bin/SER_Server,解除端口占用。

3.示例见 ./test.java





