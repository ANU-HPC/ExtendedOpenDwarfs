/**
 * Ported to OpenCL from CUDA version of
 * GPU Temporal Data Mining (Sean Ponce)
 **/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "types.h"
#include "global.h"
#include "dataio.h"

#include <include/rdtsc.h>
#include <include/common_args.h>
#include <include/lsb.h>
#include <include/portable_memory.h>

#define AOCL_ALIGNMENT 64
#define MIN_TIME_SEC 2

unsigned int numCandidates;
int episodesCulled = 0;

ubyte* h_events;
cl_mem d_events;

float* h_times;
cl_mem d_times;

int maxLevel;

ubyte* h_episodeCandidates;
cl_mem d_episodeCandidates;

float* h_episodeIntervals;
cl_mem d_episodeIntervals;

uint* h_episodeSupport;
cl_mem d_episodeSupport;

int* h_episodeIndices;
cl_mem d_episodeIndices;

uint eventSize;
uint eventType, uniqueEvents;

unsigned int h_foundCount;
cl_mem d_startRecords;
cl_uint2* h_startRecords;

cl_mem d_foundRecords;
cl_uint2* h_foundRecords;

cl_mem d_recCount;
cl_mem d_recOffSet;

cl_mem d_eventCounts;
uint* h_eventCounts;

cl_mem d_eventIndex;
cl_uint2** h_eventIndex;

FILE* dumpFile;

float* temporalConstraint;
unsigned int temporalConstraintSize;

void runTest(int argc, char** argv);

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

	h_eventIndex = (cl_uint2**) memalign(AOCL_ALIGNMENT, uniqueEvents * sizeof(cl_uint2*));
	h_eventCounts = (uint*) memalign(AOCL_ALIGNMENT, uniqueEvents * sizeof(uint));

	memset(h_eventCounts, 0, uniqueEvents * sizeof(unsigned int));

	for (long idx = 0; idx < eventSize; idx++) {
		h_eventCounts[h_events[idx] - start]++;
	}

	for (long idx = 0; idx < uniqueEvents; idx++) {
		h_eventIndex[idx] = (cl_uint2*) memalign(AOCL_ALIGNMENT, h_eventCounts[idx] * sizeof(cl_uint2));
	}

	memset(h_eventCounts, 0, uniqueEvents * sizeof(unsigned int));

	for (long idx = 0; idx < eventSize; idx++) {
		h_eventIndex[h_events[idx] - start][h_eventCounts[h_events[idx] - start]].s[0] =
			h_eventIndex[h_events[idx] - start][h_eventCounts[h_events[idx] - start]].s[1] =
				(unsigned int) idx;
		h_eventCounts[h_events[idx] - start]++;
	}
}

int compareUint2ByY(const void* v1, const void* v2)
{
	cl_uint2 i1 = *(cl_uint2*) v1;
	cl_uint2 i2 = *(cl_uint2*) v2;
	return i1.s[1] - i2.s[1];
}

cl_uint2* eliminateConflicts(cl_uint2* records, uint recordCount, uint* resultCount)
{
	cl_uint2* result = (cl_uint2*) memalign(AOCL_ALIGNMENT, recordCount * sizeof(cl_uint2));
	uint count = 0;
	uint lastRecord = 0;

	for (uint idx = 0; idx < recordCount; idx++) {
		if (records[idx].s[0] >= lastRecord) {
			result[count] = records[idx];
			count++;
			lastRecord = records[idx].s[1];
		}
	}

	*resultCount = count;
	return result;
}

void setupGpu()
{
	cl_int errcode = CL_SUCCESS;

	record_region_start("device_side_buffer_setup");

	d_events = clCreateBuffer(context, CL_MEM_READ_WRITE, eventSize * sizeof(ubyte), NULL, &errcode);
	CHKERR(errcode, "Failed to create d_events");

	d_times = clCreateBuffer(context, CL_MEM_READ_WRITE, eventSize * sizeof(float), NULL, &errcode);
	CHKERR(errcode, "Failed to create d_times");

	d_episodeCandidates = clCreateBuffer(context, CL_MEM_READ_WRITE, maxCandidates * sizeof(ubyte), NULL, &errcode);
	CHKERR(errcode, "Failed to create d_episodeCandidates");

	d_episodeIntervals = clCreateBuffer(context, CL_MEM_READ_WRITE, maxIntervals * sizeof(float), NULL, &errcode);
	CHKERR(errcode, "Failed to create d_episodeIntervals");

	d_episodeSupport = clCreateBuffer(context, CL_MEM_READ_WRITE, maxCandidates * sizeof(uint), NULL, &errcode);
	CHKERR(errcode, "Failed to create d_episodeSupport");

	d_startRecords = clCreateBuffer(context, CL_MEM_READ_WRITE, MaxRecords * sizeof(cl_uint2), NULL, &errcode);
	CHKERR(errcode, "Failed to create d_startRecords");

	d_foundRecords = clCreateBuffer(context, CL_MEM_READ_WRITE, MaxRecords * sizeof(cl_uint2), NULL, &errcode);
	CHKERR(errcode, "Failed to create d_foundRecords");

	h_startRecords = (cl_uint2*) memalign(AOCL_ALIGNMENT, MaxRecords * sizeof(cl_uint2));
	h_foundRecords = (cl_uint2*) memalign(AOCL_ALIGNMENT, MaxRecords * sizeof(cl_uint2));

	d_recCount = clCreateBuffer(context, CL_MEM_READ_WRITE, MaxRecords * sizeof(cl_uint), NULL, &errcode);
	CHKERR(errcode, "Failed to create d_recCount");

	d_recOffSet = clCreateBuffer(context, CL_MEM_READ_WRITE, MaxRecords * sizeof(cl_uint), NULL, &errcode);
	CHKERR(errcode, "Failed to create d_recOffSet");

	record_region_end(0);

	record_region_start("device_side_h2d_copy");

	errcode = clEnqueueWriteBuffer(commands, d_events, CL_TRUE, 0, eventSize, (void*) h_events, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "d_events Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	CHKERR(errcode, "Failed to enqueue d_events write buffer");

	errcode = clEnqueueWriteBuffer(commands, d_times, CL_TRUE, 0, eventSize * sizeof(float), (void*) h_times, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "d_times Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	CHKERR(errcode, "Failed to enqueue d_times write buffer");

	errcode = clEnqueueWriteBuffer(commands, d_episodeCandidates, CL_TRUE, 0, numCandidates * maxLevel * sizeof(ubyte), (void*) h_episodeCandidates, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "d_episodeCandidates Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	CHKERR(errcode, "Failed to enqueue d_episodeCandidates write buffer");

	errcode = clEnqueueWriteBuffer(commands, d_episodeIntervals, CL_TRUE, 0, numCandidates * (maxLevel - 1) * 2 * sizeof(float), (void*) h_episodeIntervals, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "d_episodeIntervals Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	CHKERR(errcode, "Failed to enqueue d_episodeIntervals write buffer");

	record_region_end(0);
}

int main(int argc, char** argv)
{
	LSB_Init(get_lsb_name(), 0);
	LSB_Set_Rparam_int("repeats_to_two_seconds", 0);
	LSB_Set_Rparam_int("num_candidates", 0);
	LSB_Set_Rparam_int("max_level", 0);
	LSB_Set_Rparam_int("event_size", 0);
	LSB_Set_Rparam_int("num_threads", 0);

	runTest(argc, argv);

	LSB_Finalize();

	return 0;
}

void runTest(int argc, char** argv)
{
	cl_program clProgram;
	cl_kernel clKernel_writeCandidates;
	cl_kernel clKernel_countCandidates;
	cl_int errcode;

	unsigned int* buff1 = NULL;
	unsigned int* buff2 = NULL;

	record_region_start("runtime_initialization");

	ocd_init(&argc, &argv, NULL);
	ocd_initCL();

	record_region_end(0);

	if (argc != 5) {
		printf("Usage: tdm_ocl <data path> <intervals path> <episodes path> <threads>\n");

		record_region_start("runtime_finalization");
		ocd_finalize();
		record_region_end(0);

		return;
	}

	record_region_start("program_build");

	clProgram = ocdBuildProgramFromFile(context, device_id, "tdm_ocl_kernel", NULL);

	record_region_end(0);

	record_region_start("kernel_creation");

	clKernel_writeCandidates = clCreateKernel(clProgram, "writeCandidates", &errcode);
	CHKERR(errcode, "Failed to create writeCandidates kernel");

	clKernel_countCandidates = clCreateKernel(clProgram, "countCandidates", &errcode);
	CHKERR(errcode, "Failed to create countCandidates kernel");

	record_region_end(0);

	size_t globalWorkSize[3];
	size_t localWorkSize[3];

	unsigned int num_threads = atoi(argv[4]);
	LSB_Set_Rparam_int("num_threads", (int) num_threads);

	record_region_start("host_input_setup");

	dumpFile = fopen("tdm-gpu-csw.txt", "w");

	loadData(argv[1], &h_events, &h_times, &eventSize, &eventType, &uniqueEvents);
	loadTemporalConstraints(argv[2], &temporalConstraint, &temporalConstraintSize);

	h_episodeIntervals = (float*) memalign(AOCL_ALIGNMENT, maxIntervals * sizeof(float));
	h_episodeCandidates = (ubyte*) memalign(AOCL_ALIGNMENT, maxCandidates * sizeof(ubyte));
	h_episodeSupport = (uint*) memalign(AOCL_ALIGNMENT, maxCandidates * sizeof(uint));

	loadCandidateEpisodes(argv[3], maxLevel);

	record_region_end(0);

	LSB_Set_Rparam_int("num_candidates", (int) numCandidates);
	LSB_Set_Rparam_int("max_level", maxLevel);
	LSB_Set_Rparam_int("event_size", (int) eventSize);

	setupGpu();

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

			errcode = clEnqueueWriteBuffer(
				commands,
				d_startRecords,
				CL_TRUE,
				0,
				h_eventCounts[currentEpisode[maxLevel - 1] - start] * sizeof(cl_uint2),
				(void*) h_eventIndex[currentEpisode[maxLevel - 1] - start],
				0,
				NULL,
				&ocdTempEvent);

			clFinish(commands);

			record_region_end(candidate_idx);

			START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "d_startRecords Copy", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(errcode, "Failed to enqueue d_startRecords write buffer");

			h_foundCount = h_eventCounts[currentEpisode[maxLevel - 1] - start];

			localWorkSize[0] = num_threads;
			localWorkSize[1] = 1;
			localWorkSize[2] = 1;

			for (int level = 2; level <= maxLevel; level++) {
				int level_idx = maxLevel - level;

				uint blockCount = (h_foundCount / num_threads) + (h_foundCount % num_threads == 0 ? 0 : 1);

				globalWorkSize[0] = blockCount * num_threads;
				globalWorkSize[1] = 1;
				globalWorkSize[2] = 1;

				record_region_start("setting_count_kernel_arguments");

				errcode = clSetKernelArg(clKernel_countCandidates, 0, sizeof(ubyte), (void*) &currentEpisode[level_idx]);
				errcode |= clSetKernelArg(clKernel_countCandidates, 1, sizeof(float), (void*) &temporalConstraint[0]);
				errcode |= clSetKernelArg(clKernel_countCandidates, 2, sizeof(float), (void*) &temporalConstraint[1]);
				errcode |= clSetKernelArg(clKernel_countCandidates, 3, sizeof(cl_mem), (void*) &d_startRecords);
				errcode |= clSetKernelArg(clKernel_countCandidates, 4, sizeof(cl_mem), (void*) &d_recCount);
				errcode |= clSetKernelArg(clKernel_countCandidates, 5, sizeof(uint), (void*) &h_foundCount);
				errcode |= clSetKernelArg(clKernel_countCandidates, 6, sizeof(cl_mem), (void*) &d_events);
				errcode |= clSetKernelArg(clKernel_countCandidates, 7, sizeof(cl_mem), (void*) &d_times);

				CHKERR(errcode, "Failed to set countCandidates kernel arguments");

				record_region_end(level);

				record_region_start("count_kernel_execution");

				errcode = clEnqueueNDRangeKernel(
					commands,
					clKernel_countCandidates,
					1,
					NULL,
					globalWorkSize,
					localWorkSize,
					0,
					NULL,
					&ocdTempEvent);

				clFinish(commands);

				record_region_end(level);

				CHKERR(errcode, "Failed to enqueue countCandidates kernel");

				START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "countCandidates kernel", ocdTempTimer)
				END_TIMER(ocdTempTimer)

				record_region_start("host_output_setup");

				buff1 = (unsigned int*) memalign(AOCL_ALIGNMENT, sizeof(unsigned int) * h_foundCount);
				buff2 = (unsigned int*) memalign(AOCL_ALIGNMENT, sizeof(unsigned int) * (h_foundCount + 1));

				if (buff1 == NULL || buff2 == NULL) {
					printf("Failed to allocate memory\n");
					return;
				}

				record_region_end(level);

				record_region_start("device_side_d2h_copy");

				errcode = clEnqueueReadBuffer(
					commands,
					d_recCount,
					CL_TRUE,
					0,
					h_foundCount * sizeof(uint),
					(void*) buff1,
					0,
					NULL,
					&ocdTempEvent);

				clFinish(commands);

				record_region_end(level);

				START_TIMER(ocdTempEvent, OCD_TIMER_D2H, "d_recCount Copy", ocdTempTimer)
				END_TIMER(ocdTempTimer)

				CHKERR(errcode, "Failed to enqueue d_recCount read buffer");

				record_region_start("host_output_setup");

				buff2[0] = 0;

				for (uint scan_idx = 0; scan_idx < h_foundCount; scan_idx++) {
					buff2[scan_idx + 1] = buff2[scan_idx] + buff1[scan_idx];
				}

				record_region_end(level);

				record_region_start("device_side_h2d_copy");

				errcode = clEnqueueWriteBuffer(
					commands,
					d_recOffSet,
					CL_TRUE,
					0,
					(h_foundCount + 1) * sizeof(cl_uint),
					(void*) buff2,
					0,
					NULL,
					&ocdTempEvent);

				clFinish(commands);

				record_region_end(level);

				START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "d_recOffset Copy", ocdTempTimer)
				END_TIMER(ocdTempTimer)

				CHKERR(errcode, "Failed to enqueue d_recOffSet write buffer");

				record_region_start("setting_write_kernel_arguments");

				errcode = clSetKernelArg(clKernel_writeCandidates, 0, sizeof(ubyte), (void*) &currentEpisode[level_idx]);
				errcode |= clSetKernelArg(clKernel_writeCandidates, 1, sizeof(float), (void*) &temporalConstraint[0]);
				errcode |= clSetKernelArg(clKernel_writeCandidates, 2, sizeof(float), (void*) &temporalConstraint[1]);
				errcode |= clSetKernelArg(clKernel_writeCandidates, 3, sizeof(cl_mem), (void*) &d_startRecords);
				errcode |= clSetKernelArg(clKernel_writeCandidates, 4, sizeof(cl_mem), (void*) &d_recOffSet);
				errcode |= clSetKernelArg(clKernel_writeCandidates, 5, sizeof(cl_mem), (void*) &d_foundRecords);
				errcode |= clSetKernelArg(clKernel_writeCandidates, 6, sizeof(uint), (void*) &h_foundCount);
				errcode |= clSetKernelArg(clKernel_writeCandidates, 7, sizeof(cl_mem), (void*) &d_events);
				errcode |= clSetKernelArg(clKernel_writeCandidates, 8, sizeof(cl_mem), (void*) &d_times);

				CHKERR(errcode, "Failed to set writeCandidates kernel arguments");

				record_region_end(level);

				record_region_start("write_kernel_execution");

				errcode = clEnqueueNDRangeKernel(
					commands,
					clKernel_writeCandidates,
					1,
					NULL,
					globalWorkSize,
					localWorkSize,
					0,
					NULL,
					&ocdTempEvent);

				clFinish(commands);

				record_region_end(level);

				CHKERR(errcode, "Failed to enqueue writeCandidates kernel");

				START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "writeCandidates kernel", ocdTempTimer)
				END_TIMER(ocdTempTimer)

				h_foundCount = buff2[h_foundCount];

				free(buff1);
				free(buff2);
				buff1 = NULL;
				buff2 = NULL;

				cl_mem temp = d_startRecords;
				d_startRecords = d_foundRecords;
				d_foundRecords = temp;
			}

			record_region_start("device_side_d2h_copy");

			errcode = clEnqueueReadBuffer(
				commands,
				d_startRecords,
				CL_TRUE,
				0,
				h_foundCount * sizeof(cl_uint2),
				(void*) h_startRecords,
				0,
				NULL,
				&ocdTempEvent);

			clFinish(commands);

			record_region_end(candidate_idx);

			START_TIMER(ocdTempEvent, OCD_TIMER_D2H, "d_startRecords Copy", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(errcode, "Failed to enqueue d_startRecords read buffer");

			record_region_start("host_output_setup");

			cl_uint2* result = eliminateConflicts(h_startRecords, h_foundCount, &h_episodeSupport[candidate_idx]);
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

	clReleaseMemObject(d_events);
	clReleaseMemObject(d_times);
	clReleaseMemObject(d_episodeCandidates);
	clReleaseMemObject(d_episodeIntervals);
	clReleaseMemObject(d_episodeSupport);
	clReleaseMemObject(d_startRecords);
	clReleaseMemObject(d_foundRecords);
	clReleaseMemObject(d_recCount);
	clReleaseMemObject(d_recOffSet);

	record_region_end(0);

	record_region_start("kernel_cleanup");

	clReleaseKernel(clKernel_countCandidates);
	clReleaseKernel(clKernel_writeCandidates);
	clReleaseProgram(clProgram);

	record_region_end(0);

	record_region_start("runtime_finalization");

	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	ocd_finalize();

	record_region_end(0);

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
}
