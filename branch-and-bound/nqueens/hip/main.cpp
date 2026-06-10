// N-queen solver for HIP
// Adapted from OpenDwarfs OpenCL N-Queens and CUDA port

#include <iostream>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <cstdlib>

#include "nqueen_cpu.h"
#include "nqueen_hip.h"

#if __has_include(<liblsb.h>)
#include <liblsb.h>
#define HAVE_LSB 1
#else
#define HAVE_LSB 0
#endif

static void usage(const char* argv0)
{
    std::cerr << "N-Queen solver for HIP\n";
    std::cerr << "Ping-Che Chen / OpenDwarfs HIP port\n\n";
    std::cerr << "Usage: " << argv0 << " [options] N\n";
    std::cerr << "\tN: board size (1 ~ 32)\n";
    std::cerr << "\t-cpu: use CPU\n";
    std::cerr << "\t-threads #: set number of HIP threads/work items\n";
    std::cerr << "\t-blocksize #: set HIP block size\n";
    std::cerr << "\t-local: accepted for OpenCL compatibility; ignored in HIP scalar port\n";
    std::cerr << "\t-noatomics: accepted for compatibility; scalar HIP port does not use atomics\n";
    std::cerr << "\t-novec: accepted for compatibility; scalar HIP port is non-vectorized\n";
    std::cerr << "\t-vec4: accepted for compatibility; ignored in HIP scalar port\n";
}

int main(int argc, char** argv)
{
    std::cerr << "N-Queen solver for HIP\n";
    std::cerr << "Ping-Che Chen / OpenDwarfs HIP port\n\n";

    if (argc < 2) {
        usage(argv[0]);
        return 0;
    }

#if HAVE_LSB
    const char* lsb_name = std::getenv("ODW_LSB_NAME");
    if (lsb_name == nullptr || lsb_name[0] == '\0') {
        lsb_name = "nqueens";
    }
    LSB_Init(lsb_name, 0);
    LSB_Set_Rparam_int("board_size", 0);
    LSB_Set_Rparam_int("j", 0);
#endif

    bool force_cpu = false;
    int threads = 0;
    int block_size = 0;
    bool local = false;
    bool noatomics = false;
    bool novec = false;
    bool use_vec4 = false;

    int start = 1;
    while (start < argc - 1) {
        if (std::strcmp(argv[start], "-cpu") == 0) {
            force_cpu = true;
        }
        else if (std::strcmp(argv[start], "-threads") == 0 && start < argc - 2) {
            threads = std::atoi(argv[start + 1]);
            start++;
        }
        else if (std::strcmp(argv[start], "-blocksize") == 0 && start < argc - 2) {
            block_size = std::atoi(argv[start + 1]);
            start++;
        }
        else if (std::strcmp(argv[start], "-local") == 0) {
            local = true;
        }
        else if (std::strcmp(argv[start], "-noatomics") == 0) {
            noatomics = true;
        }
        else if (std::strcmp(argv[start], "-novec") == 0) {
            novec = true;
        }
        else if (std::strcmp(argv[start], "-vec4") == 0) {
            use_vec4 = true;
        }
        else {
            std::cerr << "Unknown option " << argv[start] << "\n";
        }

        start++;
    }

    int board_size = std::atoi(argv[start]);
    if (board_size < 1 || board_size > 32) {
        std::cerr << "Invalid board size (only 1 ~ 32 allowed)\n";
#if HAVE_LSB
        LSB_Finalize();
#endif
        return 0;
    }

    long long solutions = 0;
    long long unique_solutions = 0;

    try {
        if (force_cpu) {
            std::clock_t start_time = std::clock();
            solutions = nqueen_cpu(board_size, &unique_solutions);
            std::clock_t end_time = std::clock();

            std::cerr << "CPU time used: "
                      << std::setprecision(3)
                      << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC
                      << "s\n";
        }
        else {
            NQueenSolverHIP nqueen(threads,
                                   block_size,
                                   local,
                                   noatomics,
                                   novec,
                                   use_vec4);

            std::cerr << "Device 0: HIP device\n";
            std::cerr << "\tUsing " << nqueen.GetThreads() << " threads\n";
            std::cerr << "\tBlock size = " << nqueen.GetBlockSize() << " threads\n";
            std::cerr << "\tUsing scalar HIP kernels\n";

#if HAVE_LSB
            LSB_Set_Rparam_int("board_size", board_size);
#endif

            solutions = nqueen.Compute(board_size, &unique_solutions);
        }
    }
    catch (const HipError& e) {
        std::cerr << e.what() << "\n";
#if HAVE_LSB
        LSB_Finalize();
#endif
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
#if HAVE_LSB
        LSB_Finalize();
#endif
        return 1;
    }

#if HAVE_LSB
    LSB_Finalize();
#endif

    std::cerr << board_size << "-queen has "
              << solutions << " solutions ("
              << unique_solutions << " unique)\n";

    return 0;
}
