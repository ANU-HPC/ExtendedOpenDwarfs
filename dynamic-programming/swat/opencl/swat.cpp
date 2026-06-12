#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "global.h"
#include "functions.h"
#include "timeRec.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "../../include/rdtsc.h"
#include "../../include/common_args.h"
#include "../../include/lsb.h"

#define AOCL_ALIGNMENT 64
#define MIN_TIME_SEC 2

#define MIN(a, b) \
	(a < b ? a : b)

static const char* get_lsb_name()
{
	const char* lsb_name = getenv("ODW_LSB_NAME");

	if (lsb_name == NULL || lsb_name[0] == '\0') {
		lsb_name = "swat";
	}

	return lsb_name;
}

static void record_region_start(const char* region)
{
	LSB_Set_Rparam_string("region", region);
	LSB_Res();
}

static void record_region_end(int id)
{
	LSB_Rec(id);
}

void usage(char* argv0)
{
	char* help =
		"\nUsage: %s queryFileName dbDataFileName [switches]\n\n"
		"    -v               :verbose (print details of sequence match)\n";
	fprintf(stderr, help, argv0);
	exit(-1);
}

char* loadSource(char* filePathName, size_t* fileSize)
{
	FILE* pfile;
	size_t tmpFileSize;
	char* fileBuffer;

	if (_deviceType == 3) {
		pfile = fopen(filePathName, "rb");
	} else {
		pfile = fopen(filePathName, "r");
	}

	if (pfile == NULL) {
		printf("Open file %s open error!\n", filePathName);
		return NULL;
	}

	fseek(pfile, 0, SEEK_END);
	tmpFileSize = ftell(pfile);

	fileBuffer = (char*) malloc(tmpFileSize);

	fseek(pfile, 0, SEEK_SET);
	fread(fileBuffer, sizeof(char), tmpFileSize, pfile);

	fclose(pfile);

	*fileSize = tmpFileSize;
	return fileBuffer;
}

int main(int argc, char** argv)
{
	LSB_Init(get_lsb_name(), 0);
	LSB_Set_Rparam_int("repeats_to_two_seconds", 0);
	LSB_Set_Rparam_int("query_size", 0);
	LSB_Set_Rparam_int("subsequence_count", 0);
	LSB_Set_Rparam_int("block_num", 0);
	LSB_Set_Rparam_int("thread_num", 0);

	record_region_start("runtime_initialization");

	ocd_init(&argc, &argv, NULL);
	ocd_initCL();

	record_region_end(0);

	if (argc < 3) {
		printf("Calculate similarities between two strings.\n");
		printf("Maximum length of each string is: %d\n", MAX_LEN);
		printf("Usage: %s query database\n", argv[0]);
		printf("or: %s query database [openPenalty extensionPenalty block#]\n", argv[0]);
		printf("openPenalty (5.0), extensionPenalty (0.5)\n");

		record_region_start("runtime_finalization");
		ocd_finalize();
		record_region_end(0);

		LSB_Finalize();
		return 1;
	}

	char queryFilePathName[255];
	char dbDataFilePathName[255];
	char dbLenFilePathName[255];

	int querySize;
	int subSequenceNum;
	int subSequenceSize;

	float openPenalty;
	float extensionPenalty;

	int nblosumWidth = 23;
	size_t blockSize = 64;
	size_t mfThreadNum;
	int blockNum = 14;

	cl_ulong maxLocalSize;
	int arraySize;

	struct timeval t1;
	struct timeval t2;
	float tmpTime;
	FILE* pfile;

	memset(&strTime, 0, sizeof(STRUCT_TIME));
	timerStart();

	openPenalty = 5.0f;
	extensionPenalty = 0.5f;

	if (argc == 6) {
		openPenalty = atof(argv[3]);
		extensionPenalty = atof(argv[4]);
		blockNum = atoi(argv[5]);
	}

	cl_program hProgram;
	cl_kernel hMatchStringKernel;
	cl_kernel hTraceBackKernel;
	cl_kernel hSetZeroKernel;

	cl_int err = CL_SUCCESS;

	cl_uint devBlockNum = 0;
	CHKERR(
		clGetDeviceInfo(
			device_id,
			CL_DEVICE_MAX_COMPUTE_UNITS,
			sizeof(cl_uint),
			&devBlockNum,
			0),
		"Error while querying CL_DEVICE_MAX_COMPUTE_UNITS.");

	if (devBlockNum == MIN(blockNum, devBlockNum)) {
		printf("Scaling blocks from %d to %d to fit on device\n", blockNum, devBlockNum);
		blockNum = devBlockNum;
	}

	mfThreadNum = blockNum * blockSize;

	CHKERR(
		clGetDeviceInfo(
			device_id,
			CL_DEVICE_LOCAL_MEM_SIZE,
			sizeof(cl_ulong),
			&maxLocalSize,
			0),
		"Error while querying CL_DEVICE_LOCAL_MEM_SIZE.");

	LSB_Set_Rparam_int("block_num", blockNum);
	LSB_Set_Rparam_int("thread_num", (int) mfThreadNum);

	record_region_start("program_build");

	hProgram = ocdBuildProgramFromFile(context, device_id, "kernels", NULL);

	int logSize = 3000;
	int i;
	size_t retSize = 0;
	char logTxt[3000];

	if (_deviceType != 3) {
		err = clGetProgramBuildInfo(
			hProgram,
			device_id,
			CL_PROGRAM_BUILD_LOG,
			logSize,
			logTxt,
			&retSize);

		for (i = 0; i < (int) retSize; i++) {
			printf("%c", logTxt[i]);
		}
	}

	CHKERR(err, "Build program error");

	record_region_end(0);

	record_region_start("kernel_creation");

	hMatchStringKernel = clCreateKernel(hProgram, "MatchStringGPUSync", &err);
	CHKERR(err, "Create MatchString kernel error");

	hTraceBackKernel = clCreateKernel(hProgram, "trace_back2", &err);
	CHKERR(err, "Create trace_back2 kernel error");

	hSetZeroKernel = clCreateKernel(hProgram, "setZero", &err);
	CHKERR(err, "Create setZero kernel error");

	record_region_end(0);

	sprintf(queryFilePathName, "%s", argv[1]);
	sprintf(dbDataFilePathName, "%s.data", argv[2]);
	sprintf(dbLenFilePathName, "%s.loc", argv[2]);

	bool verbose = false;
	int opt;

	while ((opt = getopt(argc, argv, "v")) != EOF) {
		switch (opt) {
			case 'v':
				verbose = true;
				break;
			case '?':
				usage(argv[0]);
				break;
			default:
				usage(argv[0]);
				break;
		}
	}

	char* allSequences = NULL;
	char* querySequence = NULL;
	char* subSequence = NULL;
	char* seq1 = NULL;
	char* seq2 = NULL;

	record_region_start("host_input_setup");

	posix_memalign((void**) &allSequences, AOCL_ALIGNMENT, sizeof(char) * 2 * MAX_LEN);

	if (allSequences == NULL) {
		printf("Allocate sequence buffer error!\n");
		return 1;
	}

	querySequence = allSequences;

	querySize = readQuerySequence(queryFilePathName, querySequence);
	if (querySize <= 0 || querySize > MAX_LEN) {
		printf("Query size %d is out of range (0, %d)\n", querySize, MAX_LEN);
		return 1;
	}

	encoding(querySequence, querySize);
	subSequence = allSequences + querySize;

	char* outSeq1 = new char[2 * MAX_LEN];
	char* outSeq2 = new char[2 * MAX_LEN];

	if (outSeq1 == NULL || outSeq2 == NULL) {
		printf("Allocate output sequence buffer on host error!\n");
		return 1;
	}

	int* threadNum = NULL;
	int* diffPos = NULL;

	posix_memalign((void**) &diffPos, AOCL_ALIGNMENT, sizeof(int) * 2 * MAX_LEN);
	posix_memalign((void**) &threadNum, AOCL_ALIGNMENT, sizeof(int) * 2 * MAX_LEN);

	if (threadNum == NULL || diffPos == NULL) {
		printf("Allocate location buffer on host error!\n");
		return 1;
	}

	char* pathFlag = NULL;
	char* extFlag = NULL;
	float* nGapDist = NULL;
	float* hGapDist = NULL;
	float* vGapDist = NULL;

	int maxElemNum = (MAX_LEN + 1) * (MAX_LEN + 1);

	pathFlag = new char[maxElemNum];
	extFlag = new char[maxElemNum];
	nGapDist = new float[maxElemNum];
	hGapDist = new float[maxElemNum];
	vGapDist = new float[maxElemNum];

	if (pathFlag == NULL ||
	    extFlag == NULL ||
	    nGapDist == NULL ||
	    hGapDist == NULL ||
	    vGapDist == NULL) {
		printf("Allocate DP matrices on host error!\n");
		return 1;
	}

	MAX_INFO* maxInfo = new MAX_INFO[1];
	if (maxInfo == NULL) {
		printf("Allocate maxInfo on host error!\n");
		return 1;
	}

	float* blosum62_1d = NULL;
	posix_memalign((void**) &blosum62_1d, AOCL_ALIGNMENT, sizeof(float) * 23 * 23);

	for (int ii = 0; ii < 23; ii++) {
		for (int jj = 0; jj < 23; jj++) {
			blosum62_1d[23 * ii + jj] = blosum62[ii][jj];
		}
	}

	pDBDataFile = fopen(dbDataFilePathName, "rb");
	if (pDBDataFile == NULL) {
		printf("DB data file %s open error!\n", dbDataFilePathName);
		return 1;
	}

	pDBLenFile = fopen(dbLenFilePathName, "rb");
	if (pDBLenFile == NULL) {
		printf("DB length file %s open error!\n", dbLenFilePathName);
		return 1;
	}

	fread(&subSequenceNum, sizeof(int), 1, pDBLenFile);

	record_region_end(0);

	LSB_Set_Rparam_int("query_size", querySize);
	LSB_Set_Rparam_int("subsequence_count", subSequenceNum);

	record_region_start("device_side_buffer_setup");

	cl_mem seq1D;
	cl_mem seq2D;
	cl_mem outSeq1D;
	cl_mem outSeq2D;
	cl_mem threadNumD;
	cl_mem diffPosD;
	cl_mem pathFlagD;
	cl_mem extFlagD;
	cl_mem nGapDistD;
	cl_mem hGapDistD;
	cl_mem vGapDistD;
	cl_mem maxInfoD;
	cl_mem blosum62D;
	cl_mem mutexMem;

	seq1D = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_char) * MAX_LEN, 0, &err);
	CHKERR(err, "Create seq1D memory");

	seq2D = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_char) * MAX_LEN, 0, &err);
	CHKERR(err, "Create seq2D memory");

	outSeq1D = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * MAX_LEN * 2, 0, &err);
	CHKERR(err, "Create outSeq1D memory");

	outSeq2D = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * MAX_LEN * 2, 0, &err);
	CHKERR(err, "Create outSeq2D memory");

	threadNumD = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * (2 * MAX_LEN), 0, &err);
	CHKERR(err, "Create threadNumD memory");

	diffPosD = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * (2 * MAX_LEN), 0, &err);
	CHKERR(err, "Create diffPosD memory");

	pathFlagD = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * maxElemNum, 0, &err);
	CHKERR(err, "Create pathFlagD memory");

	extFlagD = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * maxElemNum, 0, &err);
	CHKERR(err, "Create extFlagD memory");

	nGapDistD = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * maxElemNum, 0, &err);
	CHKERR(err, "Create nGapDistD memory");

	hGapDistD = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * maxElemNum, 0, &err);
	CHKERR(err, "Create hGapDistD memory");

	vGapDistD = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * maxElemNum, 0, &err);
	CHKERR(err, "Create vGapDistD memory");

	maxInfoD = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(MAX_INFO) * mfThreadNum, 0, &err);
	CHKERR(err, "Create maxInfoD memory");

	int nblosumHeight = 23;
	blosum62D = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		sizeof(cl_float) * nblosumWidth * nblosumHeight,
		0,
		&err);
	CHKERR(err, "Create blosum62D memory");

	mutexMem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), 0, &err);
	CHKERR(err, "create mutex mem error!");

	record_region_end(0);

	record_region_start("device_side_h2d_copy");

	err = clEnqueueWriteBuffer(
		commands,
		blosum62D,
		CL_TRUE,
		0,
		nblosumWidth * nblosumHeight * sizeof(cl_float),
		blosum62[0],
		0,
		NULL,
		&ocdTempEvent);

	clFinish(commands);
	record_region_end(0);

	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "SWAT Scoring Matrix Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)

	CHKERR(err, "copy blosum62 to device");

	timerEnd();
	strTime.iniTime = elapsedTime();

	int lsb_timing_repeats = 0;
	struct timeval startTime;
	struct timeval currentTime;
	struct timeval elapsed;
	gettimeofday(&startTime, NULL);

	do {
		LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

		rewind(pDBLenFile);
		rewind(pDBDataFile);
		fread(&subSequenceNum, sizeof(int), 1, pDBLenFile);

		int subSequenceNo;
		int launchNum;
		int rowNum;
		int columnNum;
		int matrixIniNum;
		int DPMatrixSize;

		for (subSequenceNo = 0; subSequenceNo < subSequenceNum; subSequenceNo++) {
			record_region_start("host_input_setup");

			timerStart();

			fread(&subSequenceSize, sizeof(int), 1, pDBLenFile);
			if (subSequenceSize <= 0 || subSequenceSize > MAX_LEN) {
				printf(
					"Size %d of subject sequence %d is out of range!\n",
					subSequenceSize,
					subSequenceNo);
				break;
			}

			fread(subSequence, sizeof(char), subSequenceSize, pDBDataFile);

			gettimeofday(&t1, NULL);

			if (subSequenceSize > querySize) {
				seq1 = subSequence;
				seq2 = querySequence;
				rowNum = subSequenceSize + 1;
				columnNum = querySize + 1;
			} else {
				seq1 = querySequence;
				seq2 = subSequence;
				rowNum = querySize + 1;
				columnNum = subSequenceSize + 1;
			}

			launchNum = rowNum + columnNum - 1;

			DPMatrixSize = preProcessing(
				rowNum,
				columnNum,
				threadNum,
				diffPos,
				matrixIniNum);

			timerEnd();
			strTime.preprocessingTime += elapsedTime();

			record_region_end(subSequenceNo);

			timerStart();

			record_region_start("device_side_buffer_fill");

			cl_event fillEvents[7];
			int zero = 0;

			err = CL_SUCCESS;
			err |= clEnqueueFillBuffer(commands, pathFlagD, &zero, sizeof(zero), 0,
				DPMatrixSize * sizeof(char), 0, NULL, &fillEvents[0]);
			err |= clEnqueueFillBuffer(commands, extFlagD, &zero, sizeof(zero), 0,
				DPMatrixSize * sizeof(char), 0, NULL, &fillEvents[1]);
			err |= clEnqueueFillBuffer(commands, nGapDistD, &zero, sizeof(zero), 0,
				matrixIniNum * sizeof(float), 0, NULL, &fillEvents[2]);
			err |= clEnqueueFillBuffer(commands, hGapDistD, &zero, sizeof(zero), 0,
				matrixIniNum * sizeof(float), 0, NULL, &fillEvents[3]);
			err |= clEnqueueFillBuffer(commands, vGapDistD, &zero, sizeof(zero), 0,
				matrixIniNum * sizeof(float), 0, NULL, &fillEvents[4]);
			err |= clEnqueueFillBuffer(commands, maxInfoD, &zero, sizeof(zero), 0,
				sizeof(MAX_INFO) * mfThreadNum, 0, NULL, &fillEvents[5]);
			err |= clEnqueueFillBuffer(commands, mutexMem, &zero, sizeof(zero), 0,
				sizeof(int), 0, NULL, &fillEvents[6]);

			clFinish(commands);

			record_region_end(subSequenceNo);

			CHKERR(err, "Zero matrices");

			record_region_start("device_side_h2d_copy");

			err = clEnqueueWriteBuffer(commands, seq1D, CL_FALSE, 0,
				(rowNum - 1) * sizeof(cl_char), seq1, 0, NULL, &ocdTempEvent);
			clFinish(commands);

			START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "SWAT Sequence Copy", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			err |= clEnqueueWriteBuffer(commands, seq2D, CL_FALSE, 0,
				(columnNum - 1) * sizeof(cl_char), seq2, 0, NULL, &ocdTempEvent);
			clFinish(commands);

			START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "SWAT Sequence Copy", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(err, "copy input sequence");

			err = clEnqueueWriteBuffer(commands, diffPosD, CL_FALSE, 0,
				launchNum * sizeof(cl_int), diffPos, 0, NULL, &ocdTempEvent);
			clFinish(commands);

			START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "SWAT Mutex Info Copy", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			err |= clEnqueueWriteBuffer(commands, threadNumD, CL_FALSE, 0,
				launchNum * sizeof(cl_int), threadNum, 0, NULL, &ocdTempEvent);
			clFinish(commands);

			START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "SWAT Mutex Info Copy", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(err, "copy diffpos and/or threadNum mutexMem info error!");

			record_region_end(subSequenceNo);

			timerEnd();
			strTime.copyTimeHostToDevice += elapsedTime();

			timerStart();

			record_region_start("setting_match_kernel_arguments");

			err = clSetKernelArg(hMatchStringKernel, 0, sizeof(cl_mem), (void*) &pathFlagD);
			err |= clSetKernelArg(hMatchStringKernel, 1, sizeof(cl_mem), (void*) &extFlagD);
			err |= clSetKernelArg(hMatchStringKernel, 2, sizeof(cl_mem), (void*) &nGapDistD);
			err |= clSetKernelArg(hMatchStringKernel, 3, sizeof(cl_mem), (void*) &hGapDistD);
			err |= clSetKernelArg(hMatchStringKernel, 4, sizeof(cl_mem), (void*) &vGapDistD);
			err |= clSetKernelArg(hMatchStringKernel, 5, sizeof(cl_mem), (void*) &diffPosD);
			err |= clSetKernelArg(hMatchStringKernel, 6, sizeof(cl_mem), (void*) &threadNumD);
			err |= clSetKernelArg(hMatchStringKernel, 7, sizeof(cl_int), (void*) &rowNum);
			err |= clSetKernelArg(hMatchStringKernel, 8, sizeof(cl_int), (void*) &columnNum);
			err |= clSetKernelArg(hMatchStringKernel, 9, sizeof(cl_mem), (void*) &seq1D);
			err |= clSetKernelArg(hMatchStringKernel, 10, sizeof(cl_mem), (void*) &seq2D);
			err |= clSetKernelArg(hMatchStringKernel, 11, sizeof(cl_int), (void*) &nblosumWidth);
			err |= clSetKernelArg(hMatchStringKernel, 12, sizeof(cl_float), (void*) &openPenalty);
			err |= clSetKernelArg(hMatchStringKernel, 13, sizeof(cl_float), (void*) &extensionPenalty);
			err |= clSetKernelArg(hMatchStringKernel, 14, sizeof(cl_mem), (void*) &maxInfoD);
			err |= clSetKernelArg(hMatchStringKernel, 15, sizeof(cl_mem), (void*) &blosum62D);
			err |= clSetKernelArg(hMatchStringKernel, 16, sizeof(cl_mem), (void*) &mutexMem);

			CHKERR(err, "Set match string argument error!");

			record_region_end(subSequenceNo);

			record_region_start("match_kernel_execution");

			err = clEnqueueNDRangeKernel(
				commands,
				hMatchStringKernel,
				1,
				NULL,
				&mfThreadNum,
				&blockSize,
				0,
				NULL,
				&ocdTempEvent);

			clFinish(commands);

			record_region_end(subSequenceNo);

			START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "SWAT Match Kernel", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(err, "Launch kernel match string error");

			timerEnd();
			strTime.matrixFillingTime += elapsedTime();

			timerStart();

			record_region_start("setting_traceback_kernel_arguments");

			err = clSetKernelArg(hTraceBackKernel, 0, sizeof(cl_mem), (void*) &pathFlagD);
			err |= clSetKernelArg(hTraceBackKernel, 1, sizeof(cl_mem), (void*) &extFlagD);
			err |= clSetKernelArg(hTraceBackKernel, 2, sizeof(cl_mem), (void*) &diffPosD);
			err |= clSetKernelArg(hTraceBackKernel, 3, sizeof(cl_mem), (void*) &seq1D);
			err |= clSetKernelArg(hTraceBackKernel, 4, sizeof(cl_mem), (void*) &seq2D);
			err |= clSetKernelArg(hTraceBackKernel, 5, sizeof(cl_mem), (void*) &outSeq1D);
			err |= clSetKernelArg(hTraceBackKernel, 6, sizeof(cl_mem), (void*) &outSeq2D);
			err |= clSetKernelArg(hTraceBackKernel, 7, sizeof(cl_mem), (void*) &maxInfoD);
			err |= clSetKernelArg(hTraceBackKernel, 8, sizeof(int), (void*) &mfThreadNum);

			CHKERR(err, "Set traceback kernel argument error!");

			record_region_end(subSequenceNo);

			record_region_start("traceback_kernel_execution");

			size_t tbGlobalSize[1] = {1};
			size_t tbLocalSize[1] = {1};

			err = clEnqueueNDRangeKernel(
				commands,
				hTraceBackKernel,
				1,
				NULL,
				tbGlobalSize,
				tbLocalSize,
				0,
				NULL,
				&ocdTempEvent);

			clFinish(commands);

			record_region_end(subSequenceNo);

			START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "SWAT Traceback Kernel", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(err, "Launch kernel trace back error");

			timerEnd();
			strTime.traceBackTime += elapsedTime();

			timerStart();

			record_region_start("device_side_d2h_copy");

			err = clEnqueueReadBuffer(
				commands,
				maxInfoD,
				CL_FALSE,
				0,
				sizeof(MAX_INFO),
				maxInfo,
				0,
				0,
				&ocdTempEvent);

			clFinish(commands);

			START_TIMER(ocdTempEvent, OCD_TIMER_D2H, "SWAT Max Info Copy", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(err, "Read maxInfo buffer error!");

			int maxOutputLen = rowNum + columnNum - 2;

			err = clEnqueueReadBuffer(
				commands,
				outSeq1D,
				CL_FALSE,
				0,
				maxOutputLen * sizeof(cl_char),
				outSeq1,
				0,
				0,
				&ocdTempEvent);

			clFinish(commands);

			START_TIMER(ocdTempEvent, OCD_TIMER_D2H, "SWAT Sequence Copy", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			err = clEnqueueReadBuffer(
				commands,
				outSeq2D,
				CL_FALSE,
				0,
				maxOutputLen * sizeof(cl_char),
				outSeq2,
				0,
				0,
				&ocdTempEvent);

			clFinish(commands);

			START_TIMER(ocdTempEvent, OCD_TIMER_D2H, "SWAT Sequence Copy", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(err, "Read output sequence error!");

			clFinish(commands);
			gettimeofday(&t2, NULL);

			timerEnd();
			strTime.copyTimeDeviceToHost += elapsedTime();

			record_region_end(subSequenceNo);

			if (verbose) {
				printf("============================================================\n");
				printf("Sequence pair %d:\n", subSequenceNo);
				int nlength = maxInfo->noutputlen;
				PrintAlignment(outSeq1, outSeq2, nlength, CHAR_PER_LINE, openPenalty, extensionPenalty);
				printf("Max alignment score (on device) is %.1f\n", maxInfo->fmaxscore);
				printf("openPenalty = %.1f, extensionPenalty = %.1f\n", openPenalty, extensionPenalty);
				printf(
					"Input sequence size, querySize: %d, subSequenceSize: %d\n",
					querySize,
					subSequenceSize);
				printf("Max position, seq1 = %d, seq2 = %d\n", maxInfo->nposi, maxInfo->nposj);
			}
		}

		lsb_timing_repeats++;
		gettimeofday(&currentTime, NULL);
		timersub(&currentTime, &startTime, &elapsed);
	} while (elapsed.tv_sec < MIN_TIME_SEC);

	tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
	pfile = fopen("../kernelTime.txt", "at");
	if (pfile != NULL) {
		fprintf(pfile, "verOpencl4:\t%.3f\n", tmpTime);
		fclose(pfile);
	}

	printTime_toStandardOutput();
	printTime_toFile();

	record_region_start("device_side_buffer_cleanup");

	clReleaseMemObject(seq1D);
	clReleaseMemObject(seq2D);
	clReleaseMemObject(outSeq1D);
	clReleaseMemObject(outSeq2D);
	clReleaseMemObject(threadNumD);
	clReleaseMemObject(diffPosD);
	clReleaseMemObject(pathFlagD);
	clReleaseMemObject(extFlagD);
	clReleaseMemObject(nGapDistD);
	clReleaseMemObject(hGapDistD);
	clReleaseMemObject(vGapDistD);
	clReleaseMemObject(maxInfoD);
	clReleaseMemObject(blosum62D);
	clReleaseMemObject(mutexMem);

	record_region_end(0);

	record_region_start("kernel_cleanup");

	clReleaseKernel(hMatchStringKernel);
	clReleaseKernel(hTraceBackKernel);
	clReleaseKernel(hSetZeroKernel);
	clReleaseProgram(hProgram);

	record_region_end(0);

	record_region_start("runtime_finalization");

	fclose(pDBLenFile);
	fclose(pDBDataFile);

	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	ocd_finalize();

	record_region_end(0);

	LSB_Finalize();

	free(allSequences);
	delete[] outSeq1;
	delete[] outSeq2;
	free(threadNum);
	free(diffPos);
	delete[] pathFlag;
	delete[] extFlag;
	delete[] nGapDist;
	delete[] hGapDist;
	delete[] vGapDist;
	delete[] maxInfo;
	free(blosum62_1d);

	return 0;
}
