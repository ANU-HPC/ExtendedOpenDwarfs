#include <hip/hip_runtime.h>

#include "../types.h"

extern "C" __global__
void countCandidatesHip(
	ubyte targetEvent,
	float minInterval,
	float maxInterval,
	uint2* startRecords,
	unsigned int* foundRecordsCount,
	uint recordCount,
	ubyte* d_events,
	float* d_times)
{
	uint tid = blockDim.x * blockIdx.x + threadIdx.x;
	uint2 ep;
	uint myIdx;
	float startTime;
	float curTime;

	unsigned int listSize = 0;

	if (tid < recordCount) {
		ep = startRecords[tid];
		myIdx = ep.x;

		curTime = startTime = d_times[myIdx];

		while (startTime - curTime < minInterval && myIdx > 0) {
			myIdx--;
			curTime = d_times[myIdx];
		}

		while (startTime - curTime <= maxInterval && myIdx > 0) {
			if (d_events[myIdx] == targetEvent) {
				listSize++;
			}

			myIdx--;
			curTime = d_times[myIdx];
		}

		foundRecordsCount[tid] = listSize;
	}
}

extern "C" __global__
void writeCandidatesHip(
	ubyte targetEvent,
	float minInterval,
	float maxInterval,
	uint2* startRecords,
	unsigned int* startOffset,
	uint2* foundRecords,
	uint recordCount,
	ubyte* d_events,
	float* d_times)
{
	uint tid = blockDim.x * blockIdx.x + threadIdx.x;
	uint2 ep;
	uint myIdx;
	uint offset;
	float startTime;
	float curTime;

	unsigned int listSize = 0;

	if (tid < recordCount) {
		ep = startRecords[tid];
		offset = startOffset[tid];
		myIdx = ep.x;

		curTime = startTime = d_times[myIdx];

		while (startTime - curTime < minInterval && myIdx > 0) {
			myIdx--;
			curTime = d_times[myIdx];
		}

		while (startTime - curTime <= maxInterval && myIdx > 0) {
			if (d_events[myIdx] == targetEvent) {
				foundRecords[offset + listSize].x = myIdx;
				foundRecords[offset + listSize].y = ep.y;
				listSize++;
			}

			myIdx--;
			curTime = d_times[myIdx];
		}
	}
}

extern "C" hipError_t tdm_launch_count_candidates_hip(
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
	hipStream_t stream)
{
	hipLaunchKernelGGL(
		countCandidatesHip,
		dim3(globalSize / localSize),
		dim3(localSize),
		0,
		stream,
		targetEvent,
		minInterval,
		maxInterval,
		startRecords,
		foundRecordsCount,
		recordCount,
		d_events,
		d_times);

	return hipGetLastError();
}

extern "C" hipError_t tdm_launch_write_candidates_hip(
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
	hipStream_t stream)
{
	hipLaunchKernelGGL(
		writeCandidatesHip,
		dim3(globalSize / localSize),
		dim3(localSize),
		0,
		stream,
		targetEvent,
		minInterval,
		maxInterval,
		startRecords,
		startOffset,
		foundRecords,
		recordCount,
		d_events,
		d_times);

	return hipGetLastError();
}
