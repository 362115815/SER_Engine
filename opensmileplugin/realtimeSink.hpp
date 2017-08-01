

#ifndef __CREALTIMESINK_HPP
#define __CREALTIMESINK_HPP

#include <core/smileCommon.hpp>
#include <core/dataSink.hpp>

#define COMPONENT_DESCRIPTION_CREALTIMESINK "This component exports data in real time."
#define COMPONENT_NAME_CREALTIMESINK "cRealtimeSink"

#undef class
class DLLEXPORT cRealtimeSink : public cDataSink {
private:
	FILE * filehandle;
	const char *filename;
	const char *instanceName;
	const char *instanceBase;
	bool disabledSink_;
	char delimChar;
	int lag;
	int flush;
	int prname;
	int append, timestamp, number, printHeader;
	
	const char * host_addr;//主机地址
	int host_port; //主机端口
	SOCKET socketClient;
	SOCKADDR_IN addrServer;
	

protected:
	SMILECOMPONENT_STATIC_DECL_PR

		virtual void fetchConfig();
	//virtual int myConfigureInstance();
	virtual int myFinaliseInstance();
	virtual int myTick(long long t);


public:
	//static sComponentInfo * registerComponent(cConfigManager *_confman);
	//static cSmileComponent * create(const char *_instname);
	SMILECOMPONENT_STATIC_DECL

		cRealtimeSink(const char *_name);

	virtual ~cRealtimeSink();
};




#endif // __CREALTIMESINK_HPP
