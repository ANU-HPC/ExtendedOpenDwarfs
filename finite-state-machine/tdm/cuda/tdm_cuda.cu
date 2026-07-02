/**
 * CUDA port of GPU Temporal Data Mining.
 **/

#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "../types.h"
#include "../global.h"
#include "../dataio.h"

#include <include/lsb.h>
#include <include/portable_memory.h>

#define AOCL_ALIGNMENT 64
#define MIN_TIME_SEC 2

unsigned int MaxRecords = 5000000;
unsigned int maxCandidates = 13000000;
unsigned int maxIntervals = (maxCandidates - 1) * 2;
unsigned int numCandidates;
int episodesCulled = 0;

ubyte* h_events;
ubyte* d_events;

float* h_times;
float* d_times;

int maxLevel;

ubyte* h_episodeCandidates;
ubyte* d_episodeCandidates;

float* h_episodeIntervals;
float* d_episodeIntervals;

uint* h_episodeSupport;
uint* d_episodeSupport;

int* h_episodeIndices;
int* d_episodeIndices;

uint eventSize;
uint eventType, uniqueEvents;

unsigned int h_foundCount;
uint2* d_startRecords;
uint2* h_startRecords;

uint2* d_foundRecords;
uint2* h_foundRecords;

uint* d_recCount;
uint* d_recOffSet;

uint* h_eventCounts;
uint2** h_eventIndex;

FILE* dumpFile;

float* temporalConstraint;
unsigned int temporalConstraintSize;

extern "C" cudaError_t tdm_launch_count_candidates_cuda(
	ubyte targetEvent,
	float minInterval,
	float maxInterval,
	uint2* startRecords,
	unsigned int* foundRecordsCount,
	uint recordCount,
	ubyte* d_events,
	float* d_times,
	size_t globalSize,
	size_t localSize,
	cudaStream_t stream);

extern "C" cudaError_t tdm_launch_write_candidates_cuda(
	ubyte targetEvent,
	float minInterval,
	float maxInterval,
	uint2* startRecords,
	unsigned int* startOffset,
	uint2* foundRecords,
	uint recordCount,
	ubyte* d_events,
	float* d_times,
	size_t globalSize,
	size_t localSize,
	cudaStream_t stream);

static const char* get_lsb_name()
{
	const char* lsb_name = getenv("ODW_LSB_NAME");

	if (lsb_name == NULL || lsb_name[0] == '\0') {
		lsb_name = "tdm";
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

static void cuda_check(cudaError_t err, const char* msg)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

static double tdm_device_working_memory_kib()
{
	size_t bytes = 0;

	bytes += (size_t) eventSize * sizeof(ubyte);
	bytes += (size_t) eventSize * sizeof(float);
	bytes += (size_t) maxCandidates * sizeof(ubyte);
	bytes += (size_t) maxIntervals * sizeof(float);
	bytes += (size_t) maxCandidates * sizeof(uint);
	bytes += (size_t) MaxRecords * sizeof(*h_startRecords);
	bytes += (size_t) MaxRecords * sizeof(*h_foundRecords);
	bytes += (size_t) MaxRecords * sizeof(uint);
	bytes += (size_t) MaxRecords * sizeof(uint);

	return (double) bytes / 1024.0;
}

static void parse_pre_separator_common_args(int argc, char** argv, int* cuda_device)
{
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--") == 0) {
			break;
		}

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
		for (int i = separator + 1; i < argc; i++) {
			(*app_argv)[(*app_argc)++] = argv[i];
		}
	} else {
		for (int i = 1; i < argc; i++) {
			(*app_argv)[(*app_argc)++] = argv[i];
		}
	}

	(*app_argv)[*app_argc] = NULL;
}

void loadCandidateEpisodes(char* filename, int& level)
{
	FILE* fp;
	char buf[256], symbol, c, *evt;
	int v, tlevel = 0, flag = 0;

	fp = fopen(filename, "r");
	level = 0;

	if (fp) {
		int idx = 0;

		while (fgets(buf, 256, fp) != NULL) {
			if (strlen(buf) == 0) {
				continue;
			}

			evt = strtok(buf, " ,.-");
			int i = 0;

			while (evt != NULL) {
				if (strlen(evt) == 1) {
					sscanf(evt, "%c", &symbol);
				} else {
					sscanf(evt, "%c%d", &c, &v);
					symbol = symbolToChar(c, v);
				}

				if (flag == 0) {
					tlevel++;
				}

				h_episodeCandidates[(idx * level) + i] = symbol;
				h_episodeIntervals[(2 * idx * (level - 1)) + i * 2 + 0] = temporalConstraint[0];
				h_episodeIntervals[(2 * idx * (level - 1)) + i * 2 + 1] = temporalConstraint[1];

				evt = strtok(NULL, " ,.-");
				i++;
			}

			if (flag == 0) {
				flag = 1;
				level = tlevel;
			}

			idx++;
		}

		numCandidates = idx;
		fclose(fp);
	} else {
		printf("Unable to read episode file\n");
		exit(1);
	}
}

void indexEvents()
{
	char start = eventType == EVENT_26 ? 'A' : '!';

	h_eventIndex = (uint2**) memalign(AOCL_ALIGNMENT, uniqueEvents * sizeof(uint2*));
	h_eventCounts = (uint*) memalign(AOCL_ALIGNMENT, uniqueEvents * sizeof(uint));

	memset(h_eventCounts, 0, uniqueEvents * sizeof(unsigned int));

	for (long idx = 0; idx < eventSize; idx++) {
		h_eventCounts[h_events[idx] - start]++;
	}

	for (long idx = 0; idx < uniqueEvents; idx++) {
		h_eventIndex[idx] = (uint2*) memalign(AOCL_ALIGNMENT, h_eventCounts[idx] * sizeof(uint2));
	}

	memset(h_eventCounts, 0, uniqueEvents * sizeof(unsigned int));

	for (long idx = 0; idx < eventSize; idx++) {
		h_eventIndex[h_events[idx] - start][h_eventCounts[h_events[idx] - start]].x =
			h_eventIndex[h_events[idx] - start][h_eventCounts[h_events[idx] - start]].y =
				(unsigned int) idx;
		h_eventCounts[h_events[idx] - start]++;
	}
}

int compareUint2ByY(const void* v1, const void* v2)
{
	uint2 i1 = *(uint2*) v1;
	uint2 i2 = *(uint2*) v2;
	return i1.y - i2.y;
}

uint2* eliminateConflicts(uint2* records, uint recordCount, uint* resultCount)
{
	uint2* result = (uint2*) memalign(AOCL_ALIGNMENT, recordCount * sizeof(uint2));
	uint count = 0;
	uint lastRecord = 0;

	for (uint idx = 0; idx < recordCount; idx++) {
		if (records[idx].x >= lastRecord) {
			result[count] = records[idx];
			count++;
			lastRecord = records[idx].y;
		}
	}

	*resultCount = count;
	return result;
}

void setupGpu(cudaStream_t stream)
{
	record_region_start("device_side_buffer_setup");

	cuda_check(cudaMalloc((void**) &d_events, eventSize * sizeof(ubyte)), "failed to allocate d_events");
	cuda_check(cudaMalloc((void**) &d_times, eventSize * sizeof(float)), "failed to allocate d_times");
	cuda_check(cudaMalloc((void**) &d_episodeCandidates, maxCandidates * sizeof(ubyte)), "failed to allocate d_episodeCandidates");
	cuda_check(cudaMalloc((void**) &d_episodeIntervals, maxIntervals * sizeof(float)), "failed to allocate d_episodeIntervals");
	cuda_check(cudaMalloc((void**) &d_episodeSupport, maxCandidates * sizeof(uint)), "failed to allocate d_episodeSupport");
	cuda_check(cudaMalloc((void**) &d_startRecords, MaxRecords * sizeof(uint2)), "failed to allocate d_startRecords");
	cuda_check(cudaMalloc((void**) &d_foundRecords, MaxRecords * sizeof(uint2)), "failed to allocate d_foundRecords");

	h_startRecords = (uint2*) memalign(AOCL_ALIGNMENT, MaxRecords * sizeof(uint2));
	h_foundRecords = (uint2*) memalign(AOCL_ALIGNMENT, MaxRecords * sizeof(uint2));

	cuda_check(cudaMalloc((void**) &d_recCount, MaxRecords * sizeof(uint)), "failed to allocate d_recCount");
	cuda_check(cudaMalloc((void**) &d_recOffSet, MaxRecords * sizeof(uint)), "failed to allocate d_recOffSet");

	record_region_end(0);

	record_region_start("device_side_h2d_copy");

	cuda_check(cudaMemcpyAsync(d_events, h_events, eventSize, cudaMemcpyHostToDevice, stream), "failed to copy d_events");
	cuda_check(cudaMemcpyAsync(d_times, h_times, eventSize * sizeof(float), cudaMemcpyHostToDevice, stream), "failed to copy d_times");
	cuda_check(cudaMemcpyAsync(
		d_episodeCandidates,
		h_episodeCandidates,
		numCandidates * maxLevel * sizeof(ubyte),
		cudaMemcpyHostToDevice,
		stream),
		"failed to copy d_episodeCandidates");
	cuda_check(cudaMemcpyAsync(
		d_episodeIntervals,
		h_episodeIntervals,
		numCandidates * (maxLevel - 1) * 2 * sizeof(float),
		cudaMemcpyHostToDevice,
		stream),
		"failed to copy d_episodeIntervals");

	cuda_check(cudaStreamSynchronize(stream), "failed to synchronize setup H2D copies");

	record_region_end(0);
}

int main(int argc, char** argv)
{
	int cuda_device = 0;
	parse_pre_separator_common_args(argc, argv, &cuda_device);

	int app_argc = 0;
	char** app_argv = NULL;
	build_app_argv(argc, argv, &app_argc, &app_argv);

	LSB_Init(get_lsb_name(), 0);
	LSB_Set_Rparam_int("repeats_to_two_seconds", 0);
	LSB_Set_Rparam_int("num_candidates", 0);
	LSB_Set_Rparam_int("max_level", 0);
	LSB_Set_Rparam_int("event_size", 0);
	LSB_Set_Rparam_int("num_threads", 0);
	LSB_Set_Rparam_int("max_records", 0);
	LSB_Set_Rparam_int("max_candidates", 0);

	record_region_start("runtime_initialization");

	cuda_check(cudaSetDevice(cuda_device), "failed to select CUDA device");
	cuda_check(cudaFree(0), "failed to initialize CUDA runtime");

	cudaStream_t stream;
	cuda_check(cudaStreamCreate(&stream), "failed to create CUDA stream");

	record_region_end(0);

	if (app_argc != 7) {
		printf("Usage: tdm_cuda <data path> <intervals path> <episodes path> <threads> <max records> <max candidates>\n");

		record_region_start("runtime_finalization");
		cuda_check(cudaStreamDestroy(stream), "failed to destroy CUDA stream");
		record_region_end(0);

		LSB_Finalize();
		free(app_argv);
		return 1;
	}

	record_region_start("kernel_creation");
	record_region_end(0);

	size_t globalWorkSize;
	size_t localWorkSize;

	unsigned int num_threads = atoi(app_argv[4]);
	MaxRecords = (unsigned int) strtoul(app_argv[5], NULL, 10);
	maxCandidates = (unsigned int) strtoul(app_argv[6], NULL, 10);
	maxIntervals = (maxCandidates - 1) * 2;

	if (MaxRecords == 0 || maxCandidates == 0) {
		printf("max records and max candidates must be > 0\n");
		LSB_Finalize();
		free(app_argv);
		return 1;
	}

	LSB_Set_Rparam_int("num_threads", (int) num_threads);
	LSB_Set_Rparam_int("max_records", (int) MaxRecords);
	LSB_Set_Rparam_int("max_candidates", (int) maxCandidates);

	record_region_start("host_input_setup");

	dumpFile = fopen("tdm-gpu-csw.txt", "w");

	loadData(app_argv[1], &h_events, &h_times, &eventSize, &eventType, &uniqueEvents);
	loadTemporalConstraints(app_argv[2], &temporalConstraint, &temporalConstraintSize);

	h_episodeIntervals = (float*) memalign(AOCL_ALIGNMENT, maxIntervals * sizeof(float));
	h_episodeCandidates = (ubyte*) memalign(AOCL_ALIGNMENT, maxCandidates * sizeof(ubyte));
	h_episodeSupport = (uint*) memalign(AOCL_ALIGNMENT, maxCandidates * sizeof(uint));

	loadCandidateEpisodes(app_argv[3], maxLevel);

	record_region_end(0);

	LSB_Set_Rparam_int("num_candidates", (int) numCandidates);
	LSB_Set_Rparam_int("max_level", maxLevel);
	LSB_Set_Rparam_int("event_size", (int) eventSize);

	printf("Working kernel memory: %.4fKiB\n", tdm_device_working_memory_kib());

	setupGpu(stream);

	record_region_start("host_input_setup");
	indexEvents();
	record_region_end(0);

	free(h_events);
	free(h_times);

	ubyte start = eventType == EVENT_26 ? 'A' : '!';

	int lsb_timing_repeats = 0;
	struct timeval startTime;
	struct timeval currentTime;
	struct timeval elapsed;

	gettimeofday(&startTime, NULL);

	do {
		LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

		for (unsigned int candidate_idx = 0; candidate_idx < numCandidates; candidate_idx++) {
			ubyte* currentEpisode = &h_episodeCandidates[candidate_idx * maxLevel];

			record_region_start("device_side_h2d_copy");

			cuda_check(cudaMemcpyAsync(
				d_startRecords,
				h_eventIndex[currentEpisode[maxLevel - 1] - start],
				h_eventCounts[currentEpisode[maxLevel - 1] - start] * sizeof(uint2),
				cudaMemcpyHostToDevice,
				stream),
				"failed to copy d_startRecords");

			cuda_check(cudaStreamSynchronize(stream), "failed to synchronize d_startRecords copy");

			record_region_end(candidate_idx);

			h_foundCount = h_eventCounts[currentEpisode[maxLevel - 1] - start];

			localWorkSize = num_threads;

			for (int level = 2; level <= maxLevel; level++) {
				int level_idx = maxLevel - level;

				uint blockCount = (h_foundCount / num_threads) + (h_foundCount % num_threads == 0 ? 0 : 1);
				globalWorkSize = blockCount * num_threads;

				record_region_start("setting_count_kernel_arguments");
				record_region_end(level);

				record_region_start("count_kernel_execution");

				cuda_check(
					tdm_launch_count_candidates_cuda(
						currentEpisode[level_idx],
						temporalConstraint[0],
						temporalConstraint[1],
						d_startRecords,
						d_recCount,
						h_foundCount,
						d_events,
						d_times,
						globalWorkSize,
						localWorkSize,
						stream),
					"failed to launch countCandidates kernel");

				cuda_check(cudaStreamSynchronize(stream), "failed to synchronize countCandidates kernel");

				record_region_end(level);

				record_region_start("host_output_setup");

				unsigned int* buff1 = (unsigned int*) memalign(AOCL_ALIGNMENT, sizeof(unsigned int) * h_foundCount);
				unsigned int* buff2 = (unsigned int*) memalign(AOCL_ALIGNMENT, sizeof(unsigned int) * (h_foundCount + 1));

				if (buff1 == NULL || buff2 == NULL) {
					printf("Failed to allocate memory\n");
					return 1;
				}

				record_region_end(level);

				record_region_start("device_side_d2h_copy");

				cuda_check(cudaMemcpyAsync(
					buff1,
					d_recCount,
					h_foundCount * sizeof(uint),
					cudaMemcpyDeviceToHost,
					stream),
					"failed to copy d_recCount");

				cuda_check(cudaStreamSynchronize(stream), "failed to synchronize d_recCount copy");

				record_region_end(level);

				record_region_start("host_output_setup");

				buff2[0] = 0;

				for (uint scan_idx = 0; scan_idx < h_foundCount; scan_idx++) {
					buff2[scan_idx + 1] = buff2[scan_idx] + buff1[scan_idx];
				}

				record_region_end(level);

				record_region_start("device_side_h2d_copy");

				cuda_check(cudaMemcpyAsync(
					d_recOffSet,
					buff2,
					(h_foundCount + 1) * sizeof(uint),
					cudaMemcpyHostToDevice,
					stream),
					"failed to copy d_recOffSet");

				cuda_check(cudaStreamSynchronize(stream), "failed to synchronize d_recOffSet copy");

				record_region_end(level);

				record_region_start("setting_write_kernel_arguments");
				record_region_end(level);

				record_region_start("write_kernel_execution");

				cuda_check(
					tdm_launch_write_candidates_cuda(
						currentEpisode[level_idx],
						temporalConstraint[0],
						temporalConstraint[1],
						d_startRecords,
						d_recOffSet,
						d_foundRecords,
						h_foundCount,
						d_events,
						d_times,
						globalWorkSize,
						localWorkSize,
						stream),
					"failed to launch writeCandidates kernel");

				cuda_check(cudaStreamSynchronize(stream), "failed to synchronize writeCandidates kernel");

				record_region_end(level);

				h_foundCount = buff2[h_foundCount];

				free(buff1);
				free(buff2);

				uint2* temp = d_startRecords;
				d_startRecords = d_foundRecords;
				d_foundRecords = temp;
			}

			record_region_start("device_side_d2h_copy");

			cuda_check(cudaMemcpyAsync(
				h_startRecords,
				d_startRecords,
				h_foundCount * sizeof(uint2),
				cudaMemcpyDeviceToHost,
				stream),
				"failed to copy d_startRecords back");

			cuda_check(cudaStreamSynchronize(stream), "failed to synchronize d_startRecords read");

			record_region_end(candidate_idx);

			record_region_start("host_output_setup");

			uint2* result = eliminateConflicts(h_startRecords, h_foundCount, &h_episodeSupport[candidate_idx]);
			free(result);

			record_region_end(candidate_idx);
		}

		lsb_timing_repeats++;
		gettimeofday(&currentTime, NULL);
		timersub(&currentTime, &startTime, &elapsed);
	} while (elapsed.tv_sec < MIN_TIME_SEC);

	record_region_start("host_output_setup");

	saveResult(
		dumpFile,
		maxLevel,
		numCandidates,
		h_episodeSupport,
		h_episodeCandidates,
		h_episodeIntervals,
		eventType);

	record_region_end(0);

	printf("Cleaning up memory...\n");

	record_region_start("device_side_buffer_cleanup");

	cudaFree(d_events);
	cudaFree(d_times);
	cudaFree(d_episodeCandidates);
	cudaFree(d_episodeIntervals);
	cudaFree(d_episodeSupport);
	cudaFree(d_startRecords);
	cudaFree(d_foundRecords);
	cudaFree(d_recCount);
	cudaFree(d_recOffSet);

	record_region_end(0);

	record_region_start("runtime_finalization");

	cuda_check(cudaStreamDestroy(stream), "failed to destroy CUDA stream");

	record_region_end(0);

	LSB_Finalize();

	free(h_startRecords);
	free(h_foundRecords);
	free(h_episodeIntervals);
	free(h_episodeCandidates);
	free(h_episodeSupport);
	free(temporalConstraint);

	if (h_eventCounts != NULL) {
		free(h_eventCounts);
	}

	if (h_eventIndex != NULL) {
		for (uint idx = 0; idx < uniqueEvents; idx++) {
			free(h_eventIndex[idx]);
		}
		free(h_eventIndex);
	}

	fclose(dumpFile);
	free(app_argv);

	return 0;
}
