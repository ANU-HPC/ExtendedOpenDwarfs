#include <cuda_runtime.h>

#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "global.h"
#include "functions.h"
#include "timeRec.h"

#include "../../include/lsb.h"

#define AOCL_ALIGNMENT 64
#define MIN_TIME_SEC 2

#define MIN(a, b) ((a) < (b) ? (a) : (b))

extern "C" cudaError_t swat_launch_match_cuda(
	char* pathFlag,
	char* extFlag,
	float* nGapDist,
	float* hGapDist,
	float* vGapDist,
	int* diffPos,
	int* threadNum,
	int rowNum,
	int columnNum,
	char* seq1,
	char* seq2,
	int blosumWidth,
	float openPenalty,
	float extensionPenalty,
	MAX_INFO* maxInfo,
	float* blosum62D,
	int* mutexMem,
	size_t mfThreadNum,
	size_t blockSize,
	cudaStream_t stream);

extern "C" cudaError_t swat_launch_traceback_cuda(
	char* pathFlag,
	char* extFlag,
	int* diffPos,
	char* seq1,
	char* seq2,
	char* outSeq1,
	char* outSeq2,
	MAX_INFO* maxInfo,
	int mfThreadNum,
	cudaStream_t stream);

static const char* get_lsb_name()
{
	const char* lsb_name = getenv("ODW_LSB_NAME");
	return (lsb_name == NULL || lsb_name[0] == '\0') ? "swat" : lsb_name;
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

static void cuda_check(cudaError_t err, const char* msg)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

static void usage(char* argv0)
{
	fprintf(stderr,
		"\nUsage: %s queryFileName dbDataFileName [openPenalty extensionPenalty block#] [-v]\n\n"
		"    -v               : verbose\n",
		argv0);
	exit(EXIT_FAILURE);
}

static void parse_pre_separator_common_args(int argc, char** argv, int* cuda_device)
{
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--") == 0)
			break;

		if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
			*cuda_device = atoi(argv[i + 1]);
			i++;
		} else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "-t") == 0) && i + 1 < argc) {
			i++;
		}
	}
}

static void build_app_argv(int argc, char** argv, int* app_argc, char*** app_argv)
{
	int separator = -1;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--") == 0) {
			separator = i;
			break;
		}
	}

	*app_argv = (char**) malloc(sizeof(char*) * (argc + 1));
	if (*app_argv == NULL) {
		fprintf(stderr, "Failed to allocate argument list\n");
		exit(EXIT_FAILURE);
	}

	(*app_argv)[0] = argv[0];
	*app_argc = 1;

	if (separator >= 0) {
		for (int i = separator + 1; i < argc; i++)
			(*app_argv)[(*app_argc)++] = argv[i];
	} else {
		for (int i = 1; i < argc; i++)
			(*app_argv)[(*app_argc)++] = argv[i];
	}

	(*app_argv)[*app_argc] = NULL;
}

int main(int argc, char** argv)
{
	int cuda_device = 0;
	parse_pre_separator_common_args(argc, argv, &cuda_device);

	int app_argc = 0;
	char** app_argv = NULL;
	build_app_argv(argc, argv, &app_argc, &app_argv);

	if (app_argc < 3)
		usage(argv[0]);

	LSB_Init(get_lsb_name(), 0);
	LSB_Set_Rparam_int("repeats_to_two_seconds", 0);
	LSB_Set_Rparam_int("query_size", 0);
	LSB_Set_Rparam_int("subsequence_count", 0);
	LSB_Set_Rparam_int("block_num", 0);
	LSB_Set_Rparam_int("thread_num", 0);

	record_region_start("runtime_initialization");

	cuda_check(cudaSetDevice(cuda_device), "failed to select CUDA device");
	cuda_check(cudaFree(0), "failed to initialize CUDA runtime");

	cudaStream_t stream;
	cuda_check(cudaStreamCreate(&stream), "failed to create CUDA stream");

	record_region_end(0);

	char queryFilePathName[255];
	char dbDataFilePathName[255];
	char dbLenFilePathName[255];

	sprintf(queryFilePathName, "%s", app_argv[1]);
	sprintf(dbDataFilePathName, "%s.data", app_argv[2]);
	sprintf(dbLenFilePathName, "%s.loc", app_argv[2]);

	float openPenalty = 5.0f;
	float extensionPenalty = 0.5f;
	int blockNum = 14;

	if (app_argc >= 6) {
		openPenalty = atof(app_argv[3]);
		extensionPenalty = atof(app_argv[4]);
		blockNum = atoi(app_argv[5]);
	}

	bool verbose = false;
	optind = 1;
	int opt;
	while ((opt = getopt(app_argc, app_argv, "v")) != EOF) {
		switch (opt) {
			case 'v':
				verbose = true;
				break;
			case '?':
			default:
				usage(argv[0]);
		}
	}

	free(app_argv);

	size_t blockSize = 64;
	int devBlockNum = 0;
	cuda_check(cudaDeviceGetAttribute(
		&devBlockNum,
		cudaDevAttrMultiProcessorCount,
		cuda_device),
		"failed to query CUDA SM count");

	if (devBlockNum == MIN(blockNum, devBlockNum)) {
		printf("Scaling blocks from %d to %d to fit on device\n", blockNum, devBlockNum);
		blockNum = devBlockNum;
	}

	size_t mfThreadNum = blockNum * blockSize;

	LSB_Set_Rparam_int("block_num", blockNum);
	LSB_Set_Rparam_int("thread_num", (int) mfThreadNum);

	record_region_start("kernel_creation");
	record_region_end(0);

	memset(&strTime, 0, sizeof(STRUCT_TIME));
	timerStart();

	record_region_start("host_input_setup");

	char* allSequences = NULL;
	posix_memalign((void**) &allSequences, AOCL_ALIGNMENT, sizeof(char) * 2 * MAX_LEN);

	if (allSequences == NULL) {
		fprintf(stderr, "Allocate sequence buffer error!\n");
		exit(EXIT_FAILURE);
	}

	char* querySequence = allSequences;
	int querySize = readQuerySequence(queryFilePathName, querySequence);

	if (querySize <= 0 || querySize > MAX_LEN) {
		fprintf(stderr, "Query size %d is out of range (0, %d)\n", querySize, MAX_LEN);
		exit(EXIT_FAILURE);
	}

	encoding(querySequence, querySize);
	char* subSequence = allSequences + querySize;

	char* outSeq1 = new char[2 * MAX_LEN];
	char* outSeq2 = new char[2 * MAX_LEN];

	int* threadNum = NULL;
	int* diffPos = NULL;
	posix_memalign((void**) &diffPos, AOCL_ALIGNMENT, sizeof(int) * 2 * MAX_LEN);
	posix_memalign((void**) &threadNum, AOCL_ALIGNMENT, sizeof(int) * 2 * MAX_LEN);

	int maxElemNum = (MAX_LEN + 1) * (MAX_LEN + 1);

	char* pathFlag = new char[maxElemNum];
	char* extFlag = new char[maxElemNum];
	float* nGapDist = new float[maxElemNum];
	float* hGapDist = new float[maxElemNum];
	float* vGapDist = new float[maxElemNum];

	MAX_INFO* maxInfo = new MAX_INFO[1];

	float* blosum62_1d = NULL;
	posix_memalign((void**) &blosum62_1d, AOCL_ALIGNMENT, sizeof(float) * 23 * 23);

	for (int i = 0; i < 23; i++) {
		for (int j = 0; j < 23; j++) {
			blosum62_1d[23 * i + j] = blosum62[i][j];
		}
	}

	pDBDataFile = fopen(dbDataFilePathName, "rb");
	if (pDBDataFile == NULL) {
		fprintf(stderr, "DB data file %s open error!\n", dbDataFilePathName);
		exit(EXIT_FAILURE);
	}

	pDBLenFile = fopen(dbLenFilePathName, "rb");
	if (pDBLenFile == NULL) {
		fprintf(stderr, "DB length file %s open error!\n", dbLenFilePathName);
		exit(EXIT_FAILURE);
	}

	int subSequenceNum = 0;
	fread(&subSequenceNum, sizeof(int), 1, pDBLenFile);

	record_region_end(0);

	LSB_Set_Rparam_int("query_size", querySize);
	LSB_Set_Rparam_int("subsequence_count", subSequenceNum);

	record_region_start("device_side_buffer_setup");

	char* seq1D = NULL;
	char* seq2D = NULL;
	char* outSeq1D = NULL;
	char* outSeq2D = NULL;
	int* threadNumD = NULL;
	int* diffPosD = NULL;
	char* pathFlagD = NULL;
	char* extFlagD = NULL;
	float* nGapDistD = NULL;
	float* hGapDistD = NULL;
	float* vGapDistD = NULL;
	MAX_INFO* maxInfoD = NULL;
	float* blosum62D = NULL;
	int* mutexMem = NULL;

	cuda_check(cudaMalloc((void**) &seq1D, sizeof(char) * MAX_LEN), "failed to allocate seq1D");
	cuda_check(cudaMalloc((void**) &seq2D, sizeof(char) * MAX_LEN), "failed to allocate seq2D");
	cuda_check(cudaMalloc((void**) &outSeq1D, sizeof(char) * MAX_LEN * 2), "failed to allocate outSeq1D");
	cuda_check(cudaMalloc((void**) &outSeq2D, sizeof(char) * MAX_LEN * 2), "failed to allocate outSeq2D");
	cuda_check(cudaMalloc((void**) &threadNumD, sizeof(int) * 2 * MAX_LEN), "failed to allocate threadNumD");
	cuda_check(cudaMalloc((void**) &diffPosD, sizeof(int) * 2 * MAX_LEN), "failed to allocate diffPosD");
	cuda_check(cudaMalloc((void**) &pathFlagD, sizeof(char) * maxElemNum), "failed to allocate pathFlagD");
	cuda_check(cudaMalloc((void**) &extFlagD, sizeof(char) * maxElemNum), "failed to allocate extFlagD");
	cuda_check(cudaMalloc((void**) &nGapDistD, sizeof(float) * maxElemNum), "failed to allocate nGapDistD");
	cuda_check(cudaMalloc((void**) &hGapDistD, sizeof(float) * maxElemNum), "failed to allocate hGapDistD");
	cuda_check(cudaMalloc((void**) &vGapDistD, sizeof(float) * maxElemNum), "failed to allocate vGapDistD");
	cuda_check(cudaMalloc((void**) &maxInfoD, sizeof(MAX_INFO) * mfThreadNum), "failed to allocate maxInfoD");
	cuda_check(cudaMalloc((void**) &blosum62D, sizeof(float) * 23 * 23), "failed to allocate blosum62D");
	cuda_check(cudaMalloc((void**) &mutexMem, sizeof(int)), "failed to allocate mutexMem");

	record_region_end(0);

	record_region_start("device_side_h2d_copy");

	cuda_check(cudaMemcpyAsync(
		blosum62D,
		blosum62_1d,
		sizeof(float) * 23 * 23,
		cudaMemcpyHostToDevice,
		stream),
		"failed to copy BLOSUM matrix");

	cuda_check(cudaStreamSynchronize(stream), "failed to synchronize BLOSUM copy");

	record_region_end(0);

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

		for (int subSequenceNo = 0; subSequenceNo < subSequenceNum; subSequenceNo++) {
			record_region_start("host_input_setup");

			timerStart();

			int subSequenceSize = 0;
			fread(&subSequenceSize, sizeof(int), 1, pDBLenFile);

			if (subSequenceSize <= 0 || subSequenceSize > MAX_LEN) {
				fprintf(stderr,
					"Size %d of subject sequence %d is out of range!\n",
					subSequenceSize,
					subSequenceNo);
				break;
			}

			fread(subSequence, sizeof(char), subSequenceSize, pDBDataFile);

			struct timeval t1;
			struct timeval t2;
			gettimeofday(&t1, NULL);

			char* seq1;
			char* seq2;
			int rowNum;
			int columnNum;

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

			int launchNum = rowNum + columnNum - 1;
			int matrixIniNum = 0;

			int DPMatrixSize = preProcessing(
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

			cuda_check(cudaMemsetAsync(pathFlagD, 0, DPMatrixSize * sizeof(char), stream), "failed to clear pathFlagD");
			cuda_check(cudaMemsetAsync(extFlagD, 0, DPMatrixSize * sizeof(char), stream), "failed to clear extFlagD");
			cuda_check(cudaMemsetAsync(nGapDistD, 0, matrixIniNum * sizeof(float), stream), "failed to clear nGapDistD");
			cuda_check(cudaMemsetAsync(hGapDistD, 0, matrixIniNum * sizeof(float), stream), "failed to clear hGapDistD");
			cuda_check(cudaMemsetAsync(vGapDistD, 0, matrixIniNum * sizeof(float), stream), "failed to clear vGapDistD");
			cuda_check(cudaMemsetAsync(maxInfoD, 0, sizeof(MAX_INFO) * mfThreadNum, stream), "failed to clear maxInfoD");
			cuda_check(cudaMemsetAsync(mutexMem, 0, sizeof(int), stream), "failed to clear mutexMem");
			cuda_check(cudaStreamSynchronize(stream), "failed to synchronize buffer fill");

			record_region_end(subSequenceNo);

			record_region_start("device_side_h2d_copy");

			cuda_check(cudaMemcpyAsync(seq1D, seq1, (rowNum - 1) * sizeof(char), cudaMemcpyHostToDevice, stream), "failed to copy seq1");
			cuda_check(cudaMemcpyAsync(seq2D, seq2, (columnNum - 1) * sizeof(char), cudaMemcpyHostToDevice, stream), "failed to copy seq2");
			cuda_check(cudaMemcpyAsync(diffPosD, diffPos, launchNum * sizeof(int), cudaMemcpyHostToDevice, stream), "failed to copy diffPos");
			cuda_check(cudaMemcpyAsync(threadNumD, threadNum, launchNum * sizeof(int), cudaMemcpyHostToDevice, stream), "failed to copy threadNum");
			cuda_check(cudaStreamSynchronize(stream), "failed to synchronize H2D copies");

			record_region_end(subSequenceNo);

			timerEnd();
			strTime.copyTimeHostToDevice += elapsedTime();

			timerStart();

			record_region_start("setting_match_kernel_arguments");
			record_region_end(subSequenceNo);

			record_region_start("match_kernel_execution");

			cuda_check(
				swat_launch_match_cuda(
					pathFlagD,
					extFlagD,
					nGapDistD,
					hGapDistD,
					vGapDistD,
					diffPosD,
					threadNumD,
					rowNum,
					columnNum,
					seq1D,
					seq2D,
					23,
					openPenalty,
					extensionPenalty,
					maxInfoD,
					blosum62D,
					mutexMem,
					mfThreadNum,
					blockSize,
					stream),
				"failed to launch SWAT match kernel");

			cuda_check(cudaStreamSynchronize(stream), "failed to synchronize SWAT match kernel");

			record_region_end(subSequenceNo);

			timerEnd();
			strTime.matrixFillingTime += elapsedTime();

			timerStart();

			record_region_start("setting_traceback_kernel_arguments");
			record_region_end(subSequenceNo);

			record_region_start("traceback_kernel_execution");

			cuda_check(
				swat_launch_traceback_cuda(
					pathFlagD,
					extFlagD,
					diffPosD,
					seq1D,
					seq2D,
					outSeq1D,
					outSeq2D,
					maxInfoD,
					(int) mfThreadNum,
					stream),
				"failed to launch SWAT traceback kernel");

			cuda_check(cudaStreamSynchronize(stream), "failed to synchronize SWAT traceback kernel");

			record_region_end(subSequenceNo);

			timerEnd();
			strTime.traceBackTime += elapsedTime();

			timerStart();

			record_region_start("device_side_d2h_copy");

			cuda_check(cudaMemcpyAsync(maxInfo, maxInfoD, sizeof(MAX_INFO), cudaMemcpyDeviceToHost, stream), "failed to copy maxInfo");

			int maxOutputLen = rowNum + columnNum - 2;

			cuda_check(cudaMemcpyAsync(outSeq1, outSeq1D, maxOutputLen * sizeof(char), cudaMemcpyDeviceToHost, stream), "failed to copy outSeq1");
			cuda_check(cudaMemcpyAsync(outSeq2, outSeq2D, maxOutputLen * sizeof(char), cudaMemcpyDeviceToHost, stream), "failed to copy outSeq2");

			cuda_check(cudaStreamSynchronize(stream), "failed to synchronize D2H copies");

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
				printf("Input sequence size, querySize: %d, subSequenceSize: %d\n", querySize, subSequenceSize);
				printf("Max position, seq1 = %d, seq2 = %d\n", maxInfo->nposi, maxInfo->nposj);
			}

			(void) t1;
			(void) t2;
		}

		lsb_timing_repeats++;
		gettimeofday(&currentTime, NULL);
		timersub(&currentTime, &startTime, &elapsed);
	} while (elapsed.tv_sec < MIN_TIME_SEC);

	printTime_toStandardOutput();
	printTime_toFile();

	record_region_start("device_side_buffer_cleanup");

	cudaFree(seq1D);
	cudaFree(seq2D);
	cudaFree(outSeq1D);
	cudaFree(outSeq2D);
	cudaFree(threadNumD);
	cudaFree(diffPosD);
	cudaFree(pathFlagD);
	cudaFree(extFlagD);
	cudaFree(nGapDistD);
	cudaFree(hGapDistD);
	cudaFree(vGapDistD);
	cudaFree(maxInfoD);
	cudaFree(blosum62D);
	cudaFree(mutexMem);

	record_region_end(0);

	record_region_start("runtime_finalization");

	fclose(pDBLenFile);
	fclose(pDBDataFile);
	cudaStreamDestroy(stream);

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
