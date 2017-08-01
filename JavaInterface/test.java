import java.io.IOException;

public class test{

	public static void main(String []args)throws IOException
	{
		SER_Server ser_server=new SER_Server();
		ser_server.start_engine("/home/emo/SER/SER_Server_Release",6666); //start SRR server 
		ser_server.post_request("/home/emo/SER/JavaInterface/018_sadness_utterance_2022.wav"); //post recogniton request;
		String rt=ser_server.get_result();//get recognition result
		System.out.println(rt);
		ser_server.stop_engine();
		
	}
}