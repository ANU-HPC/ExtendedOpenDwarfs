#include <hip/hip_runtime.h>

typedef struct {
	int nposi, nposj;
	int nmaxpos;
	float fmaxscore;
	int noutputlen;
} MAX_INFO;

#define PATH_END 0
#define COALESCED_OFFSET 32

__device__ void barrier_hip_lock_based(
	int localID,
	int goalValue,
	volatile int* mutexMem)
{
	__syncthreads();
	__threadfence();

	if (localID == 0) {
		atomicAdd((int*) mutexMem, 1);
		__threadfence();

		while (atomicAdd((int*) mutexMem, 0) < goalValue) {
		}
	}

	__syncthreads();
}

__global__ void MatchStringGPUSyncHip(
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
	volatile int* mutexMem)
{
	int npos, ntablepos, tid;
	int npreposngap, npreposhgap, npreposvgap;
	int nLocalID = threadIdx.x;
	int blockNum = gridDim.x;
	int blockSize = blockDim.x;
	int totalThreadNum = blockSize * blockNum;
	int threadid = blockIdx.x * blockDim.x + threadIdx.x;

	int launchNo;
	int launchNum = rowNum + columnNum - 1;
	int indexi1 = -1;
	int indexj1 = 0;
	int indexi, indexj;
	int startPos = 2 * COALESCED_OFFSET;
	int noffset = 0;

	float fdist;
	float fdistngap, fdisthgap, fdistvgap;
	float ext_dist;
	float fmaxdist;

	for (launchNo = 2; launchNo < launchNum; launchNo++) {
		if (launchNo < rowNum) {
			indexi1++;
		} else if (launchNo == rowNum) {
			indexi1++;
			noffset = 1;
		} else {
			indexj1++;
		}

		for (tid = threadid; tid < threadNum[launchNo]; tid += totalThreadNum) {
			indexi = indexi1 - tid;
			indexj = indexj1 + tid;

			npos = startPos + tid;

			npreposhgap = npos - diffPos[launchNo];
			npreposvgap = npreposhgap - 1;
			npreposngap = npreposvgap - diffPos[launchNo - 1];

			ntablepos = seq1[indexi] * blosumWidth + seq2[indexj];
			fdist = blosum62D[ntablepos];

			fmaxdist = nGapDist[npreposngap];
			fdistngap = fmaxdist + fdist;

			ext_dist = hGapDist[npreposhgap] - extensionPenalty;
			fdisthgap = nGapDist[npreposhgap] - openPenalty;

			if (fdisthgap <= ext_dist) {
				fdisthgap = ext_dist;
				extFlag[npreposhgap] = 1;
			}

			ext_dist = vGapDist[npreposvgap] - extensionPenalty;
			fdistvgap = nGapDist[npreposvgap] - openPenalty;

			if (fdistvgap <= ext_dist) {
				fdistvgap = ext_dist;
				pathFlag[npreposvgap] += 8;
			}

			fdistngap = (fdistngap < 0.0f) ? 0.0f : fdistngap;
			fdisthgap = (fdisthgap < 0.0f) ? 0.0f : fdisthgap;
			fdistvgap = (fdistvgap < 0.0f) ? 0.0f : fdistvgap;

			hGapDist[npos] = fdisthgap;
			vGapDist[npos] = fdistvgap;

			if (fdistngap >= fdisthgap && fdistngap >= fdistvgap) {
				fmaxdist = fdistngap;
				pathFlag[npos] = 2;
			} else if (fdisthgap >= fdistngap && fdisthgap >= fdistvgap) {
				fmaxdist = fdisthgap;
				pathFlag[npos] = 1;
			} else {
				fmaxdist = fdistvgap;
				pathFlag[npos] = 3;
			}

			nGapDist[npos] = fmaxdist;

			if (fmaxdist <= 0.00000001f) {
				pathFlag[npos] = PATH_END;
			}

			if (maxInfo[threadid].fmaxscore < fmaxdist) {
				maxInfo[threadid].nposi = indexi + 1;
				maxInfo[threadid].nposj = indexj + 1;
				maxInfo[threadid].nmaxpos = npos;
				maxInfo[threadid].fmaxscore = fmaxdist;
			}
		}

		barrier_hip_lock_based(nLocalID, (launchNo - 1) * blockNum, mutexMem);
		startPos += diffPos[launchNo + 1] + noffset;
	}
}

__global__ void trace_back2_hip(
	char* str_npathflagp,
	char* str_nExtFlagp,
	int* ndiffpos,
	char* instr1D,
	char* instr2D,
	char* outstr1,
	char* outstr2,
	MAX_INFO* strMaxInfop,
	int mfThreadNum)
{
	int i, j;
	int npos, maxPos, nlen;
	int npathflag;
	int nlaunchno;
	float maxScore;

	maxPos = 0;
	maxScore = strMaxInfop[0].fmaxscore;

	for (i = 1; i < mfThreadNum; i++) {
		if (maxScore < strMaxInfop[i].fmaxscore) {
			maxPos = i;
			maxScore = strMaxInfop[i].fmaxscore;
		}
	}

	npos = strMaxInfop[maxPos].nmaxpos;
	npathflag = str_npathflagp[npos] & 0x3;
	nlen = 0;

	i = strMaxInfop[maxPos].nposi;
	j = strMaxInfop[maxPos].nposj;
	nlaunchno = i + j;

	while (1) {
		if (npathflag == 3) {
			outstr1[nlen] = 23;
			outstr2[nlen] = instr2D[j - 1];
			nlen++;
			j--;

			npos = npos - ndiffpos[nlaunchno] - 1;
			nlaunchno--;
		} else if (npathflag == 1) {
			outstr1[nlen] = instr1D[i - 1];
			outstr2[nlen] = 23;
			nlen++;
			i--;

			npos = npos - ndiffpos[nlaunchno];
			nlaunchno--;
		} else if (npathflag == 2) {
			outstr1[nlen] = instr1D[i - 1];
			outstr2[nlen] = instr2D[j - 1];
			nlen++;
			i--;
			j--;

			npos = npos - ndiffpos[nlaunchno] - ndiffpos[nlaunchno - 1] - 1;
			nlaunchno = nlaunchno - 2;
		} else {
			return;
		}

		int nExtFlag = str_npathflagp[npos] / 4;

		if (npathflag == 3 && (nExtFlag == 2 || nExtFlag == 3)) {
			npathflag = 3;
		} else if (npathflag == 1 && str_nExtFlagp[npos] == 1) {
			npathflag = 1;
		} else {
			npathflag = str_npathflagp[npos] & 0x3;
		}

		if (i == 0 || j == 0) {
			break;
		}

		if (npathflag == PATH_END) {
			break;
		}
	}

	i--;
	j--;

	while (i >= 0) {
		outstr1[nlen] = instr1D[i];
		outstr2[nlen] = 23;
		nlen++;
		i--;
	}

	while (j >= 0) {
		outstr1[nlen] = 23;
		outstr2[nlen] = instr2D[j];
		nlen++;
		j--;
	}

	strMaxInfop[0] = strMaxInfop[maxPos];
	strMaxInfop[0].noutputlen = nlen;
}

extern "C" hipError_t swat_launch_match_hip(
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
	hipStream_t stream)
{
	size_t blockNum = mfThreadNum / blockSize;

	hipLaunchKernelGGL(
		MatchStringGPUSyncHip,
		dim3(blockNum),
		dim3(blockSize),
		0,
		stream,
		pathFlag,
		extFlag,
		nGapDist,
		hGapDist,
		vGapDist,
		diffPos,
		threadNum,
		rowNum,
		columnNum,
		seq1,
		seq2,
		blosumWidth,
		openPenalty,
		extensionPenalty,
		maxInfo,
		blosum62D,
		mutexMem);

	return hipGetLastError();
}

extern "C" hipError_t swat_launch_traceback_hip(
	char* pathFlag,
	char* extFlag,
	int* diffPos,
	char* seq1,
	char* seq2,
	char* outSeq1,
	char* outSeq2,
	MAX_INFO* maxInfo,
	int mfThreadNum,
	hipStream_t stream)
{
	hipLaunchKernelGGL(
		trace_back2_hip,
		dim3(1),
		dim3(1),
		0,
		stream,
		pathFlag,
		extFlag,
		diffPos,
		seq1,
		seq2,
		outSeq1,
		outSeq2,
		maxInfo,
		mfThreadNum);

	return hipGetLastError();
}
