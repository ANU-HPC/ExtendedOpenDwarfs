#ifndef NQUEEN_CUDA_H
#define NQUEEN_CUDA_H

#include <cstddef>
#include <stdexcept>
#include <string>

class CudaError : public std::runtime_error {
public:
    CudaError(const std::string& msg, int line);
};

class NQueenSolverCUDA {
public:
    NQueenSolverCUDA(int threads,
                     int block_size,
                     bool force_local,
                     bool force_no_atomics,
                     bool force_no_vec,
                     bool force_vec4);

    ~NQueenSolverCUDA();

    long long Compute(int board_size, long long* unique);

    int GetThreads() const { return m_nThreads; }
    int GetBlockSize() const { return m_nBlockSize; }

    bool AtomicsEnabled() const { return false; }
    bool VectorizationEnabled() const { return false; }

private:
    void InitDevice(int requested_threads, int requested_block_size);
    void AllocateBuffers(int max_pitch);
    void FreeBuffers();

    int m_nThreads;
    int m_nBlockSize;
    int m_nMaxWorkItems;
    int m_nLastTotalSize;

    bool m_bForceLocal;
    bool m_bForceNoAtomics;
    bool m_bForceNoVectorization;
    bool m_bForceVec4;

    unsigned int* d_ParamBuffer;
    unsigned int* d_ResultBuffer;
    unsigned int* d_ForbiddenBuffer;
};

#endif
