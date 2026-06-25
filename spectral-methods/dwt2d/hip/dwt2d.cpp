#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <getopt.h>
#include <malloc.h>

#include "lsb.h"

#define THREADS 256
#define MIN_TIME_SEC 2
#define DIVANDRND(a, b) ((((a) % (b)) != 0) ? ((a) / (b) + 1) : ((a) / (b)))

#define CUDA_CHECK(call) do { \
    hipError_t err__ = (call); \
    if (err__ != hipSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err__)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#include "dwt2d_kernel.h"

struct dwt {
    char * srcFilename;
    char * outFilename;
    unsigned char *srcImg;
    int pixWidth;
    int pixHeight;
    int components;
    int dwtLvls;
};

static struct option longopts[] = {
    {"level",       1, 0, 'l'},
    {"97",          0, 0, '9'},
    {"53",          0, 0, '5'},
    {"write-visual",0, 0, 'w'},
    {"help",        0, 0, 'h'},
    {0,0,0,0}
};

static void usage() {
    printf("dwt [options] src_img.rgb <out_img.dwt>\n"
           "  -l, --level              DWT level, default 3\n"
           "  -9, --97                 9/7 transform\n"
           "  -5, --53                 5/3 transform\n"
           "  -w, --write-visual       write output in visual order\n");
}

static const char* get_lsb_name()
{
    const char* lsb_name = getenv("ODW_LSB_NAME");
    return (lsb_name && lsb_name[0]) ? lsb_name : "dwt2d";
}

static std::string read_token(FILE* fp)
{
    std::string tok;
    int c;

    do {
        c = fgetc(fp);
        if (c == '#') {
            while (c != '\n' && c != EOF) c = fgetc(fp);
        }
    } while (c != EOF && isspace(c));

    while (c != EOF && !isspace(c)) {
        tok.push_back((char)c);
        c = fgetc(fp);
    }

    return tok;
}

static bool loadPNM(const char* filename, int& width, int& height, int& channels, unsigned char** data)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        return false;
    }

    std::string magic = read_token(fp);
    if (magic == "P6") {
        channels = 3;
    } else if (magic == "P5") {
        channels = 1;
    } else {
        fclose(fp);
        return false;
    }

    width = atoi(read_token(fp).c_str());
    height = atoi(read_token(fp).c_str());
    int max_colour = atoi(read_token(fp).c_str());

    if (width <= 0 || height <= 0 || max_colour != 255) {
        fclose(fp);
        return false;
    }

    size_t bytes = (size_t)width * (size_t)height * (size_t)channels;
    *data = (unsigned char*)malloc(bytes);
    if (*data == NULL) {
        fclose(fp);
        return false;
    }

    size_t got = fread(*data, 1, bytes, fp);
    fclose(fp);

    if (got != bytes) {
        free(*data);
        *data = NULL;
        return false;
    }

    return true;
}

static void checksum_component(const char* label, int* component, int samplesNum)
{
    int* host = (int*)malloc((size_t)samplesNum * sizeof(int));
    if (host == NULL) {
        fprintf(stderr, "Unable to allocate checksum buffer for %s\n", label);
        exit(EXIT_FAILURE);
    }

    LSB_Set_Rparam_string("region", "device_side_d2h_copy");
    LSB_Res();
    CUDA_CHECK(hipMemcpy(host, component, (size_t)samplesNum * sizeof(int), hipMemcpyDeviceToHost));
    CUDA_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);

    double checksum = 0.0;
    double abs_checksum = 0.0;

    for (int i = 0; i < samplesNum; i++) {
        double v = (double)host[i];
        checksum += v * (double)(i + 1);
        abs_checksum += std::fabs(v);
    }

    printf("DWT2D_CHECKSUM component=%s values=%d value=%0.17e abs=%0.17e\n",
           label, samplesNum, checksum, abs_checksum);

    free(host);
}

static void copySrcToComponents(int* d_r, int* d_g, int* d_b, unsigned char* h_src, int width, int height)
{
    int pixels = width * height;
    int alignedSize = DIVANDRND(pixels, THREADS) * THREADS;

    unsigned char* d_src = NULL;

    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();
    CUDA_CHECK(hipMalloc((void**)&d_src, (size_t)pixels * 3 * sizeof(unsigned char)));
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "device_side_h2d_copy");
    LSB_Res();
    CUDA_CHECK(hipMemcpy(d_src, h_src, (size_t)pixels * 3 * sizeof(unsigned char), hipMemcpyHostToDevice));
    CUDA_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "setting_c_CopySrcToComponents_kernel_arguments");
    LSB_Res();
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "c_CopySrcToComponents_kernel");
    LSB_Res();
    c_CopySrcToComponents<<<DIVANDRND(alignedSize, THREADS), THREADS>>>(d_r, d_g, d_b, d_src, pixels);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);

    CUDA_CHECK(hipFree(d_src));
}

static void copySrcToComponent(int* d_c, unsigned char* h_src, int width, int height)
{
    int pixels = width * height;
    int alignedSize = DIVANDRND(pixels, THREADS) * THREADS;

    unsigned char* d_src = NULL;

    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();
    CUDA_CHECK(hipMalloc((void**)&d_src, (size_t)pixels * sizeof(unsigned char)));
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "device_side_h2d_copy");
    LSB_Res();
    CUDA_CHECK(hipMemcpy(d_src, h_src, (size_t)pixels * sizeof(unsigned char), hipMemcpyHostToDevice));
    CUDA_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "setting_c_CopySrcToComponent_kernel_arguments");
    LSB_Res();
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "c_CopySrcToComponent_kernel");
    LSB_Res();
    c_CopySrcToComponent<<<DIVANDRND(alignedSize, THREADS), THREADS>>>(d_c, d_src, pixels);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);

    CUDA_CHECK(hipFree(d_src));
}

static void launchFDWT53Kernel(int WIN_SX, int WIN_SY, int* in, int* out, int sx, int sy)
{
    const int steps = (sy / (15 * WIN_SY)) + ((sy % (15 * WIN_SY)) ? 1 : 0);
    int gx = (sx / WIN_SX) + ((sx % WIN_SX) ? 1 : 0);
    int gy = (sy / (WIN_SY * steps)) + ((sy % (WIN_SY * steps)) ? 1 : 0);

    static bool printed_sliding_info = false;
    if (!printed_sliding_info) {
        printf("sliding steps = %d , gx = %d , gy = %d\n", steps, gx, gy);
        printed_sliding_info = true;
    }

    LSB_Set_Rparam_string("region", "setting_kl_fdwt53Kernel_kernel_arguments");
    LSB_Res();
    LSB_Rec(0);

    size_t shared_bytes = (size_t)WIN_SX * (size_t)WIN_SY * 3 * sizeof(int);

    LSB_Set_Rparam_string("region", "kl_fdwt53Kernel_kernel");
    LSB_Res();
    cl_fdwt53Kernel<<<dim3(gx, gy), dim3(WIN_SX, 1), shared_bytes>>>(
        in, out, sx, sy, steps, WIN_SX, WIN_SY);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);
}

static void memCopy(int* dest, int* src, size_t sx, size_t sy)
{
    LSB_Set_Rparam_string("region", "device_side_d2d_copy");
    LSB_Res();
    CUDA_CHECK(hipMemcpy(dest, src, sx * sy * sizeof(int), hipMemcpyDeviceToDevice));
    CUDA_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);
}

static void fdwt53(int* in, int* out, int sizeX, int sizeY, int levels)
{
    LSB_Set_Rparam_int("dwt_level", levels);

    if (sizeX >= 960) {
        launchFDWT53Kernel(192, 8, in, out, sizeX, sizeY);
    } else if (sizeX >= 480) {
        launchFDWT53Kernel(128, 8, in, out, sizeX, sizeY);
    } else if (sizeX >= 240) {
        launchFDWT53Kernel(64, 8, in, out, sizeX, sizeY);
    } else {
        launchFDWT53Kernel(8, 8, in, out, sizeX, sizeY);
    }

    if (levels > 1) {
        const int llSizeX = (sizeX / 2) + ((sizeX % 2) ? 1 : 0);
        const int llSizeY = (sizeY / 2) + ((sizeY % 2) ? 1 : 0);
        memCopy(in, out, llSizeX, llSizeY);
        fdwt53(in, out, llSizeX, llSizeY, levels - 1);
    }
}

static int nStage2dDWT(int* in, int* out, int* backup, int pixWidth, int pixHeight, int stages, bool forward)
{
    static bool printed_dwt_stage_info = false;
    if (!printed_dwt_stage_info) {
        printf("\n*** %d stages of 2D forward DWT:\n", stages);
        printed_dwt_stage_info = true;
    }

    if (forward) {
        fdwt53(in, out, pixWidth, pixHeight, stages);
    }

    return 0;
}

static void processDWT(struct dwt *d, int forward, int writeVisual)
{
    int pixels = d->pixWidth * d->pixHeight;
    int componentSize = pixels * sizeof(int);

    int *c_r_out = NULL, *c_g_out = NULL, *c_b_out = NULL;
    int *backup = NULL;
    int *c_r = NULL, *c_g = NULL, *c_b = NULL;

    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();
    CUDA_CHECK(hipMalloc((void**)&c_r_out, componentSize));
    CUDA_CHECK(hipMalloc((void**)&backup, componentSize));

    if (d->components == 3) {
        CUDA_CHECK(hipMalloc((void**)&c_g_out, componentSize));
        CUDA_CHECK(hipMalloc((void**)&c_b_out, componentSize));
        CUDA_CHECK(hipMalloc((void**)&c_r, componentSize));
        CUDA_CHECK(hipMalloc((void**)&c_g, componentSize));
        CUDA_CHECK(hipMalloc((void**)&c_b, componentSize));
    } else {
        CUDA_CHECK(hipMalloc((void**)&c_r, componentSize));
    }
    LSB_Rec(0);

    if (d->components == 3) {
        copySrcToComponents(c_r, c_g, c_b, d->srcImg, d->pixWidth, d->pixHeight);

        printf("Working kernel memory: %fKiB\n", (componentSize * 2) / 1024.0);

        int lsb_timing_repeats = 0;
        struct timeval startTime, currentTime, elapsedTime;
        gettimeofday(&startTime, NULL);

        do {
            LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

            nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
            nStage2dDWT(c_g, c_g_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
            nStage2dDWT(c_b, c_b_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);

            lsb_timing_repeats++;
            gettimeofday(&currentTime, NULL);
            timersub(&currentTime, &startTime, &elapsedTime);
        } while (elapsedTime.tv_sec < MIN_TIME_SEC);

        checksum_component("r", c_r_out, pixels);
        checksum_component("g", c_g_out, pixels);
        checksum_component("b", c_b_out, pixels);

        CUDA_CHECK(hipFree(c_g_out));
        CUDA_CHECK(hipFree(c_b_out));
        CUDA_CHECK(hipFree(c_g));
        CUDA_CHECK(hipFree(c_b));
    } else {
        copySrcToComponent(c_r, d->srcImg, d->pixWidth, d->pixHeight);
        printf("Working kernel memory: %fKiB\n", (componentSize * 2) / 1024.0);
        nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        checksum_component("r", c_r_out, pixels);
    }

    CUDA_CHECK(hipFree(c_r_out));
    CUDA_CHECK(hipFree(backup));
    CUDA_CHECK(hipFree(c_r));
}

int main(int argc, char **argv)
{
    int optindex = 0;
    char ch;

    int dwtLvls = 3;
    int forward = 1;
    int dwt97 = 0;
    int writeVisual = 0;

    while ((ch = getopt_long(argc, argv, "l:D:95wh", longopts, &optindex)) != -1) {
        switch (ch) {
        case 'l':
            dwtLvls = atoi(optarg);
            break;
        case '9':
            dwt97 = 1;
            break;
        case '5':
            dwt97 = 0;
            break;
        case 'w':
            writeVisual = 1;
            break;
        case 'h':
            usage();
            return 0;
        case '?':
            return -1;
        default:
            usage();
            return -1;
        }
    }

    argc -= optind;
    argv += optind;

    if (argc == 0) {
        printf("Please supply src file name\n");
        usage();
        return -1;
    }

    LSB_Init(get_lsb_name(), 0);
    LSB_Set_Rparam_int("repeats_to_two_seconds", 0);
    LSB_Set_Rparam_int("dwt_level", 0);

    LSB_Set_Rparam_string("region", "runtime_initialization");
    LSB_Res();
    CUDA_CHECK(hipFree(NULL));
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "host_side_setup");
    LSB_Res();

    struct dwt *d = (struct dwt *)malloc(sizeof(struct dwt));
    if (d == NULL) {
        return EXIT_FAILURE;
    }

    d->srcImg = NULL;
    d->dwtLvls = dwtLvls;
    d->srcFilename = strdup(argv[0]);

    if (argc == 1) {
        d->outFilename = (char *)malloc(strlen(d->srcFilename) + 5);
        strcpy(d->outFilename, d->srcFilename);
        strcpy(d->outFilename + strlen(d->srcFilename), ".dwt");
    } else {
        d->outFilename = strdup(argv[1]);
    }

    if (!loadPNM(d->srcFilename, d->pixWidth, d->pixHeight, d->components, &d->srcImg)) {
        printf("Error: Couldn't load image!\n");
        LSB_Finalize();
        return EXIT_FAILURE;
    }

    printf("\nSource file:\t\t%s\n", d->srcFilename);
    printf(" Dimensions:\t\t%dx%d\n", d->pixWidth, d->pixHeight);
    printf(" DWT levels:\t\t%d\n", d->dwtLvls);
    printf(" 9/7 transform:\t\t%d\n", dwt97);

    LSB_Rec(0);

    if (dwt97 == 1) {
        fprintf(stderr, "DWT 9/7 path is not implemented in this port; use -5.\n");
        LSB_Finalize();
        return EXIT_FAILURE;
    }

    processDWT(d, forward, writeVisual);

    LSB_Set_Rparam_string("region", "runtime_finalization");
    LSB_Res();
    CUDA_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);

    LSB_Finalize();

    free(d->srcImg);
    free(d->srcFilename);
    free(d->outFilename);
    free(d);

    return 0;
}
