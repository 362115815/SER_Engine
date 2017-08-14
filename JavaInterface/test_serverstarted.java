import java.io.IOException;

public class test_serverstarted{

	public static void main(String []args)throws IOException
	{
		SER_Server ser_server=new SER_Server();
       // ser_server.start_engine("/home/ubuntu/emoSpeechWork/SER/SER_Server_Release",7777); //start SRR server
        ser_server.get_connect(7777);
		ser_server.post_request("/home/ubuntu/emoSpeechWork/SER/SER_Server_Release/test_wav/output_segment_0003_done.wav"); //post recogniton request;
		String rt=ser_server.get_result();//get recognition result
		System.out.println(rt);
		ser_server.stop_engine();

	}
}
