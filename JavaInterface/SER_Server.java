import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
public class SER_Server{

   private Socket socket;
   private  BufferedReader in;
   private  PrintWriter out;
   private boolean server_started;
   private Process p;
   public SER_Server()
   {
      server_started=false;
      p=null;
   }


  private int get_connect(int port) throws UnknownHostException,IOException
   {
        System.out.println("connecting to server...");

        InetAddress addr = InetAddress.getByName("localhost");   
        try{

        socket=new Socket(addr,port);
        }
        catch(UnknownHostException e)
        {
          e.printStackTrace();
          return -1;
        }
        catch(IOException e)
        {
          e.printStackTrace();
          return -1;
        }

         System.out.println("connected.");

        return 0;
   }

   public String get_result()throws IOException
   {
    System.out.println("getting result");

    InputStream is=socket.getInputStream();
    String info =null;
    BufferedReader br=new BufferedReader(new InputStreamReader(is));
    if((info=br.readLine())!=null)
    {
        System.out.println("got");
       // System.out.println(info);

    }
    return info;
   }

   public int post_request(String wav_path)throws IOException
   {
      System.out.println("posting request:"+wav_path);
      OutputStream os=socket.getOutputStream();
      PrintWriter pw=new PrintWriter(os);
      pw.write(wav_path);
      pw.flush();
      System.out.println("posted");


    return 0;


   }

   public int start_engine(String work_dir,int port)throws IOException
   {

      long startMili,endMili;
        System.out.println("starting SER Server..."); 
        String cmd=work_dir+"/bin/SER_Server" +" "+work_dir+ " " +port;
         p = Runtime.getRuntime().exec(cmd);
 
	   InputStream is=p.getInputStream();
	   BufferedReader reader=new BufferedReader(new InputStreamReader(is));
	
    	String s=null;
      boolean flag=false;
     startMili=System.currentTimeMillis();// 当前时间对应的毫秒数
     Long elapseMili = 0L;
    	while(!flag && (s=reader.readLine())!=null && elapseMili<10000)
	  {

      System.out.println(s);
      if(s.indexOf("SER Engine is listening")!=-1)
      {
        flag=true;
      }
      elapseMili=System.currentTimeMillis()-startMili;
      //System.out.println(elapseMili);
	  }
    if(flag)
    {
      server_started=true;
      System.out.println("SER Server started."); 
    }
    else
    {
      System.out.println("fail to start SER Server."); 
      return -1;
    }

    get_connect(port);

    return 0;
   }

   public int stop_engine()throws IOException
   {
      if(socket.isConnected())
      {
        socket.close();

      }
      if(server_started)
      {
        //kill server
        String cmd="pkill SER_Server";
        Runtime.getRuntime().exec(cmd);
      }
      return 0;
   }


}


