/*F***************************************************************************
*
* openSMILE - the Munich open source Multimedia Interpretation by
* Large-scale Extraction toolkit
*
* This file is part of openSMILE.
*
* openSMILE is copyright (c) by audEERING GmbH. All rights reserved.
*
* See file "COPYING" for details on usage rights and licensing terms.
* By using, copying, editing, compiling, modifying, reading, etc. this
* file, you agree to the licensing terms in the file COPYING.
* If you do not agree to the licensing terms,
* you must immediately destroy all copies of this file.
*
* THIS SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO EXPRESS,
* IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT LIMITATION, WARRANTIES OF
* MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, ANY WARRANTY AGAINST
* INTERFERENCE WITH YOUR ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE
* OR NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL FULFILL
* ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST PASS THIS
* DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
* NEITHER TUM NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE LIABLE FOR ANY
* DAMAGES RELATED TO THE SOFTWARE OR THIS LICENSE AGREEMENT, INCLUDING
* DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL DAMAGES, TO THE
* MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT LEGAL THEORY IT IS BASED ON.
* ALSO, YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE
* THE SOFTWARE OR DERIVATIVE WORKS.
*
* Main authors: Florian Eyben, Felix Weninger,
*         Martin Woellmer, Bjoern Schuller
*
* Copyright (c) 2008-2013,
*   Institute for Human-Machine Communication,
*   Technische Universitaet Muenchen, Germany
*
* Copyright (c) 2013-2015,
*   audEERING UG (haftungsbeschraenkt),
*   Gilching, Germany
*
* Copyright (c) 2016,
*   audEERING GmbH,
*   Gilching Germany
***************************************************************************E*/


/*  openSMILE component:

Comma Separated Value file output (CSV)

*/


#include<WinSock2.h>
#pragma comment(lib,"ws2_32.lib")
#include <iocore/realtimeSink.hpp>



#define MODULE "cRealtimeSink"

/*Library:
sComponentInfo * registerMe(cConfigManager *_confman) {
cDataSink::registerComponent(_confman);
}
*/

SMILECOMPONENT_STATICS(cRealtimeSink)

//sComponentInfo * cRealtimeSink::registerComponent(cConfigManager *_confman)
SMILECOMPONENT_REGCOMP(cRealtimeSink)
{
	SMILECOMPONENT_REGCOMP_INIT
		scname = COMPONENT_NAME_CREALTIMESINK;
	sdescription = COMPONENT_DESCRIPTION_CREALTIMESINK;

	// we inherit cDataSink configType and extend it:
	SMILECOMPONENT_INHERIT_CONFIGTYPE("cDataSink")

	SMILECOMPONENT_IFNOTREGAGAIN_BEGIN
	ct->setField("filename", "The CSV file to write to. An empty filename (or '?' as the filename) disables this sink component.", "smileoutput.csv");
	ct->setField("delimChar", "The column delimiter character to use (usually ',' or ';') (NOTE: use '<space>' or '<tab>' for these special characters respectively)", ';');
	ct->setField("lag", "output data <lag> frames behind", 0, 0, 0);
	ct->setField("append", "1 = append to an existing file, or create a new file; 0 = overwrite an existing file, or create a new file", 0);
	ct->setField("timestamp", "1 = print a timestamp attribute for each output frame (1/0 = yes/no)", 1);
	// for compatibility with arffSink
	ct->setField("frameTime", "(same as 'timestamp') 1 = print a timestamp attribute for each output frame (1/0 = yes/no)", 1);
	ct->setField("number", "1 = print an instance number (= frameIndex) attribute for each output frame (1/0 = yes/no)", 1);
	// for compatibility with arffSink
	ct->setField("frameIndex", "(same as 'number') 1 = print an instance number (= frameIndex) attribute for each output frame (1/0 = yes/no)", 1);
	ct->setField("printHeader", "1 = print a header line as the first line in the CSV file. This line contains the attribute names separated by the delimiter character.", 1);
	ct->setField("flush", "1 = flush data to file after every line written (might give low performance for small lines!).", 0);
	ct->setField("instanceBase", "if not empty, print instance name attribute <instanceBase_Nr>", (const char*)NULL);
	ct->setField("instanceName", "if not empty, print instance name attribute <instanceName>", (const char*)NULL);
	ct->setField("hostAddress", "the host address that you want to send real time data to ", (const char*)NULL);
	ct->setField("hostPort", "the host port to send real time data to ",6666);
	SMILECOMPONENT_IFNOTREGAGAIN_END

	SMILECOMPONENT_MAKEINFO(cRealtimeSink);
}

SMILECOMPONENT_CREATE(cRealtimeSink)

//-----

cRealtimeSink::cRealtimeSink(const char *_name) :
	cDataSink(_name),
	filehandle(NULL),
	filename(NULL),
	printHeader(0),
	delimChar(';'),
	instanceName(NULL),
	instanceBase(NULL),
	prname(0),
	disabledSink_(false),
	host_addr(NULL),
	host_port(6666)
{
}

void cRealtimeSink::fetchConfig()
{
	cDataSink::fetchConfig();

	/*
	filename = getStr("filename");
	if (filename == NULL || *filename == 0 || (*filename == '?' && *(filename + 1) == 0)) {
		SMILE_IMSG(2, "No filename given, disabling this sink component.");
		disabledSink_ = true;
		errorOnNoOutput_ = 0;
	}
	delimChar = getChar("delimChar");
	lag = getInt("lag");
	append = getInt("append");
	if (append) { SMILE_IDBG(3, "append to file is enabled"); }
	printHeader = getInt("printHeader");
	if (printHeader) { SMILE_IDBG(3, "printing header with feature names"); }

	number = getInt("number");
	if (isSet("frameIndex")) {
		number = getInt("frameIndex");
	}
	if (number) { SMILE_IDBG(3, "printing of frameIndex (number) to file is enabled"); }

	timestamp = getInt("timestamp");
	if (isSet("frameTime")) {
		timestamp = getInt("frameTime");
	}
	if (timestamp) { SMILE_IDBG(3, "printing timestamp attribute (index 1) enabled"); }

	flush = getInt("flush");

	if (isSet("instanceBase")) {
		instanceBase = getStr("instanceBase");
		prname = 2;
	}
	if (isSet("instanceName")) {  // instance name overrides instance base
		instanceName = getStr("instanceName");
		prname = 1;
	}

	host_addr = getStr("hostAddress");
	if (host_addr==NULL)
	{
		SMILE_IMSG(1, "No host_addr given, disabling this sink component.");
		disabledSink_ = true;
		errorOnNoOutput_ = 0;
	}
	else
	{
		SMILE_IMSG(1, "host address : %s",host_addr);
	}

	if (isSet("hostPort"))
	{
		host_port = getInt("hostPort");
	}
	SMILE_IMSG(1, "host port : %d",host_port);
	*/
	host_addr = getStr("hostAddress");
	if (host_addr == NULL)
	{
		SMILE_IMSG(1, "No host_addr given, disabling this sink component.");
		disabledSink_ = true;
		errorOnNoOutput_ = 0;
	}
	else
	{
		SMILE_IMSG(1, "host address : %s", host_addr);
	}

	if (isSet("hostPort"))
	{
		host_port = getInt("hostPort");
	}
}

/*
int cRealtimeSink::myConfigureInstance()
{
int ret=1;
ret *= cDataSink::myConfigureInstance();
// ....
//return ret;
}
*/


int cRealtimeSink::myFinaliseInstance()
{
	int ap = 0;
	if (disabledSink_) {
		return 1;
	}
	int ret = cDataSink::myFinaliseInstance();
	if (ret == 0) return 0;
	
	//connect to host
	WSAStartup(MAKEWORD(2, 2), &WSADATA());
	socketClient = socket(AF_INET, SOCK_STREAM, 0);
	addrServer.sin_addr.S_un.S_addr = inet_addr(host_addr);
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(host_port);
	int rt = connect(socketClient, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));

	if (rt!=0)
	{
		COMP_ERR("Error fail to connect to server, socket error code :%d", WSAGetLastError());
		ret = 0;
	}
	
	else 
	{
		SMILE_IMSG(1, "Server connected: %s:%d", host_addr,host_port);
		ret = 1;
	}
	

	return ret;
}


int cRealtimeSink::myTick(long long t)
{
	//char * buffer="fffff";
	//send(socketClient, buffer, strlen(buffer) + 1, 0);
	
	cVector *vec = reader_->getFrameRel(lag);
	SMILE_IDBG(4, "tick # %i, sending data to server:(lag=%i) (vec = %ld):",
		t, lag, long(vec));
	if (vec == NULL) {
		return 0;
	}
	long vi = vec->tmeta->vIdx;
	double tm = vec->tmeta->time;
	// now print the vector:
	int i;
	char buffer[50];
	for (i = 0; i<vec->N ; i++) {
		sprintf(buffer, "%e", vec->dataF[i]);
		send(socketClient, buffer, strlen(buffer) + 1, 0);
	}

	if (flush) {
		//flush();
	}

	nWritten_++;
	
	// tick success
	return 1;
}


cRealtimeSink::~cRealtimeSink()
{
	if (filehandle != NULL) {
		fclose(filehandle);
		filehandle = NULL;
	}
}

