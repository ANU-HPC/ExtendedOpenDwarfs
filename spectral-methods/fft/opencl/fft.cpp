//
// Consolidated OpenCL FFT benchmark for Extended OpenDwarfs.
// Derived from Eric Bainville OpenCLfft2 sources and the EOD benchmark wrapper.
//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <sys/time.h>

#include <CL/cl.h>

#include "common_args.h"
#include "rdtsc.h"
#include "lsb.h"

#ifdef WIN32
#define snprintf _snprintf
#endif

// Check and report OpenCL errors
// Copyright 2011, Eric Bainville
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#ifdef WIN32
#define snprintf _snprintf
#endif

namespace clfft {

#define CLFFT_CHECK_STATUS(status) clfft::checkStatus(status,__FILE__,__LINE__)

/** Check OpenCL status value, and print error message if not success.
    Use CLFFT_CHECK_STATUS(status) to call this function.

    @param status is the OpenCL value to check.
    @param filename,line are used in message.

    @return TRUE if status is CL_SUCCESS, and FALSE otherwise. */
bool checkStatus(cl_int status,const char * filename,int line);

} // namespace


// Light encapsulation of OpenCL events
// Copyright 2011, Eric Bainville
#include <vector>
#define CLFFT_CHECK_EVENT(e) CLFFT_CHECK_STATUS((e).getStatus())

namespace clfft {

class Context;
class EventVector;

// Encapsulates one OpenCL event and the status of its creation.
class Event
{
public:

  // Constructor. Initialized to event 0 (invalid).
  inline Event() : mEvent(0), mStatus(CL_INVALID_EVENT) { }

  // Copy
  inline Event(const Event & e) : mEvent(0) { assign(e.mEvent); mStatus = e.mStatus; }

  // = operator
  inline Event & operator = (const Event & e) { assign(e.mEvent); mStatus = e.mStatus; return *this; }

  // Destructor, release the event.
  inline ~Event() { assign(0); }

  // Check if valid (non 0).
  inline bool isValid() const { return (mEvent != 0); }

  // Get OpenCL status returned by the function creating the event
  inline cl_int getStatus() const { return mStatus; }

private:

  // Constructor: takes ownership of E without incrementing reference count.
  // E may be 0. STATUS is the status returned by the function creating E.
  inline Event(cl_event e,cl_int status) : mEvent(e), mStatus(status) { }

  // Special case for E=0 (to return errors).
  inline explicit Event(cl_int status) : mEvent(0), mStatus(status) { }

  // Special case for STATUS=CL_SUCCESS
  inline explicit Event(cl_event e) : mEvent(e), mStatus(CL_SUCCESS) { }

  // Access event
  inline operator cl_event () const { return mEvent; }

  // Assign value E, release previous event if any, and retain E if not 0. E may be 0.
  inline void assign(cl_event e)
  {
    if (e == mEvent) return; // Nothing to do
    if (mEvent != 0) { clReleaseEvent(mEvent); mEvent = 0; }
    if (e != 0) { mEvent = e; clRetainEvent(mEvent); }
  }

  // Encapsulated event, friends only
  cl_event mEvent;
  // Status returned when the event was created (normally CL_SUCCESS if event is valid)
  cl_int mStatus;

  friend class clfft::Context;
  friend class clfft::EventVector;
}; // class Event

// Encapsulates one vector of valid OpenCL events
class EventVector
{
public:

  // Constructor. Initialize with the given events. Events are retained.
  inline EventVector() { }
  inline EventVector(Event & e1) { append(e1); }
  inline EventVector(Event & e1,Event & e2) { append(e1); append(e2); }
  inline EventVector(Event & e1,Event & e2,Event & e3) { append(e1); append(e2); append(e3); }

  // Copy constructor
  inline EventVector(const EventVector & v)
  {
    size_t n = v.mEvents.size();
    for (size_t i=0;i<n;i++)
    {
      cl_event e = v.mEvents[i];
      clRetainEvent(e);
      mEvents.push_back(e);
    }
  }

  // = operator
  inline EventVector & operator = (const EventVector & v)
  {
    if (&v != this)
    {
      clear();
      size_t n = v.mEvents.size();
      for (size_t i=0;i<n;i++)
      {
        cl_event e = v.mEvents[i];
        clRetainEvent(e);
        mEvents.push_back(e);
      }
    }
    return *this;
  }

  // Destructor. Release the events.
  ~EventVector() { clear(); }

  // Append one event to the vector. Ignore if invalid. Otherwise the event is retained.
  inline void append(Event & e)
  {
    cl_event ce = (cl_event)e;
    if (ce == 0) return; // Invalid, ignore
    clRetainEvent(ce);
    mEvents.push_back(ce);
  }

  // Clear the vector. Release the events.
  inline void clear()
  {
    size_t n = mEvents.size();
    for (size_t i=0;i<n;i++) clReleaseEvent(mEvents[i]);
    mEvents.clear();
  }

private:

  // Access
  inline cl_uint size() const { return (cl_uint)mEvents.size(); }
  inline const cl_event * events() const { if (mEvents.empty()) return 0; else return &(mEvents[0]); }

  // Encapsulated events, friends only
  std::vector<cl_event> mEvents;

  friend class clfft::Context;
};

} // namespace


// OpenCL FFT library
// Copyright 2011, Eric Bainville
#include <string>
#include <vector>
#include <CL/cl.h>
namespace clfft {

enum RealType {
  FLOAT_REAL_TYPE = 0,
  DOUBLE_REAL_TYPE,
  NB_CLFFT_REAL_TYPES
};

enum Direction {
  FORWARD_DIRECTION = -1,
  INVERSE_DIRECTION = 1
};

class Context
{
public:

  /** Create a new instance of the class, using the given OpenCL context.
      The object manages internally command queues and compile kernels for
      all devices attached to the context.

      @param context is the OpenCl context, we acquire a reference to it.
      @param realType is the real type.
      @param msgLen is the allocated size in MSG, may be 0.
      @param msg receives error messages on failure, may be 0.

      @return a non 0 instance on success, 0 otherwise. */
  static Context * create(cl_context context,RealType realType,std::string & errorMsg);

  /** Destructor. */
  virtual ~Context();

  /** Get number of devices and command queues attached to the creation context.

      @return number of devices/command queues. */
  int getNDevices() const;

  /** Get real type.

      @return one of XXX_REAL_TYPE. */
  int getRealType() const;

  /** Get real type size.
  
      @return size of real type (bytes). */
  size_t getRealTypeSize() const;

  /** Get OpenCL context.

      @return the OpenCL context passed at creation. */
  cl_context getOpenCLContext() const;

  /** Call clFinish on command queue.

      @param device is the device index (0 = first device of context, etc.).

      @return an OpenCL status. */
  cl_int finish(int device);

  /** Enqueue a barrier in command queue.
   *
      @param device is the device index (0 = first device of context, etc.).

      @return an OpenCL status. */
  cl_int enqueueBarrier(int device);

  /** Create a complex buffer.

      @param flags are the CL_MEM_XXX flags passed to clCreateBuffer.
      @param n is the size of the buffer in complex numbers.
      @param hostPtr is the host pointer passed to clCreateBuffer, may be 0.

      @return a valid cl_mem object on success, and 0 otherwise. */
  cl_mem createComplexBuffer(cl_mem_flags flags,size_t n,void * hostPtr);

  /** Enqueue a buffer read.

      @param device is the device index (0 = first device of context, etc.).
      @param buffer,blocking_read,offset,cb,ptr are arguments of clReadBuffer.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueRead(int device,cl_mem buffer,cl_bool blocking_read,size_t offset,size_t cb,void * ptr,const EventVector & waitList);

  /** Enqueue a buffer write.

      @param device is the device index (0 = first device of context, etc.).
      @param buffer,blocking_write,offset,cb,ptr are arguments of clWriteBuffer.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueWrite(int device,cl_mem buffer,cl_bool blocking_write,size_t offset,size_t cb,const void * ptr,const EventVector & waitList);

  /** Enqueue buffer copy.

      @param device is the device index (0 = first device of context, etc.).
      @param buffer,blocking_write,offset,cb,ptr are arguments of clWriteBuffer.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueCopy(int device,cl_mem src,cl_mem dst,size_t src_offset,size_t dst_offset,size_t cb,const EventVector & waitList);

  /** Enqueue kernel to unpack and pad P real values into a buffer of N complex values.

      @param device is the device index (0 = first device of context, etc.).
      @param n is the size of the DFT, must be a power of 2.
      @param batch is the number of parallel DFT-N to process.
      @param p is the number of real values in IN.
      @param in is the input OpenCL buffer, P*BATCH real numbers.
      @param out is the output OpenCL buffer, N*BATCH complex numbers.
      @param wg is the desired workgroup size.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueUnpackReal1D(int device,size_t n,size_t batch,size_t p,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList);

  /** Enqueue kernel to pad P complex values into a buffer of N complex values.

      @param device is the device index (0 = first device of context, etc.).
      @param n is the size of the DFT, must be a power of 2.
      @param batch is the number of parallel DFT-N to process.
      @param p is the number of complex values in IN.
      @param in is the input OpenCL buffer, P*BATCH complex numbers.
      @param out is the output OpenCL buffer, N*BATCH complex numbers.
      @param wg is the desired workgroup size.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueUnpackComplex1D(int device,size_t n,size_t batch,size_t p,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList);

  /** Enqueue one radix-R kernel, one step of the computation of K parallel DFT-N.
  
      @param device is the device index (0 = first device of context, etc.).
      @param n is the size of the DFT, must be a power of 2.
      @param batch is the number of parallel DFT-N to process.
      @param p is the length of already transformed sequences in IN. RADIX*P is the length of transformed sequences in OUT.
      @param radix is the radix: 2, 4, or 8.
      @param direction is FORWARD_DIRECTION or INVERSE_DIRECTION.
      @param in is the input OpenCL buffer, N*BATCH complex numbers.
      @param out is the output OpenCL buffer, N*BATCH complex numbers.
      @param wg is the desired workgroup size.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueRadixRKernel(int device,size_t n,size_t batch,size_t p,size_t radix,Direction direction,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList);

private:

  // Constructor
  Context();

  enum Kernels {
    UNPACK_REAL1D_KERNEL = 0,
    UNPACK_COMPLEX1D_KERNEL,
    FFT_RADIX2_KERNEL,
    FFT_RADIX4_KERNEL,
    FFT_RADIX8_KERNEL,
    FFT_RADIX16_KERNEL,
    NB_KERNELS
  };

  // Functions used to setup kernel args. KERNELID is one of XXX_KERNEL.
  void clearArgs(int kernelID) { mKernelIndex[kernelID] = 0; }
  template <class T> cl_int pushArg(int kernelID,const T & x)
  {
    cl_int status = clSetKernelArg(mKernel[kernelID],mKernelIndex[kernelID]++,sizeof(T),&x);
    if (!CLFFT_CHECK_STATUS(status)) return status;
    return CL_SUCCESS;
  }
  // Run 1D or 2D kernel on device.
  Event enqueueKernel(int device,int kernelID,size_t nx,size_t ny,size_t wg,const EventVector & waitList);

  RealType mRealType; // Real type used
  size_t mRealTypeSize; // Size of the real type (bytes)
  cl_context mContext; // OpenCL context
  cl_program mProgram; // Program
  std::vector<cl_kernel> mKernel; // Kernels
  std::vector<cl_device_id> mDevice; // Devices in context
  std::vector<cl_command_queue> mQueue; // One command queue per context device
  std::vector<size_t> mMaxWorkGroupSize; // Max workgroup size for each device
  std::vector<int> mKernelIndex; // Current number of args set in each kernel
};

} // namespace


//
// Test functions
// Copyright 2011, Eric Bainville
// Modified by Beau Johnston Sep 2017.
// All rights reserved.
//
#include <stdio.h>
#include <math.h>
#include <CL/cl.h>

// Return wallclock time in seconds. (origin is arbitrary)
double getRealTime();

// Return random number in [0,1[
double rnd();

// Create an OpenCL context including all GPU devices on the first platform
// providing GPU devices.  Return a valid OpenCL context on success, and 0 otherwise.
cl_context createGPUContext();

// Initialize X[N] with random values in [-1,+1]
template <typename REAL> void rand(size_t n,REAL * x)
{
  for (size_t i=0;i<n;i++) x[i] = (REAL)(2.0*rnd()-1.0);
}

// Initialize X[N] with ones as the real component, and zeros in the imaginary part
template <typename REAL> void ones(size_t n,REAL * x)
{
  for (size_t i=0;i<n;i++){
      x[i*2] = (REAL)(1.0);
      x[i*2+1] = (REAL)(0.0);
  }
}

// Initialize X[N] with zeros as the real component, and zeros in the imaginary part
template <typename REAL> void zeros(size_t n,REAL * x)
{
  for (size_t i=0;i<n;i++){
      x[i*2] = (REAL)(0.0);
      x[i*2+1] = (REAL)(0.0);
  }
}

// Initialize X[N] with a sin wave as the real component and zeros in the imaginary part
template <typename REAL> void sine(size_t n,REAL * x)
{
    int i = 0;
    for (REAL z = (REAL)(-10.0); z < (REAL)(10.0); z+= (REAL)(20.0/n)) {
        x[i*2] = (REAL)sin(z);
        x[i*2+1] = (REAL)(0.0);
        i++;
    }
}

// Return RMSE between X[N] and Y[N]
template <typename REAL> double rmse(size_t n,const REAL * x,const REAL * y)
{
  double s = 0;
  for (size_t i=0;i<n;i++)
    {
      double d = (REAL)x[i] - (REAL)y[i];
      s += d*d;
    }
  return sqrt(s/(double)n);
}

template <typename REAL> void dumpRealArray(size_t n,const REAL * x)
{
  for (size_t i=0;i<n;i++)
    {
      printf("  %2d: %10f\n",(int)i,(double)x[2*i]);
    }
}

// Dump X[2*N] as complex array
template <typename REAL> void dumpComplexArray(size_t n,const REAL * x)
{
  for (size_t i=0;i<n;i++)
    {
      printf("  %2d: %10f,%10f\n",(int)i,(double)x[2*i],(double)x[2*i+1]);
    }
}

// Dump X[2*N] and X_REF[2*N] as complex arrays, and show differences
template <typename REAL> void dumpComplexArray(size_t n,const REAL * x,const REAL * x_ref)
{
  for (size_t i=0;i<n;i++)
    {
      double d0 = (REAL)x_ref[2*i] - (REAL)x[2*i];
      double d1 = (REAL)x_ref[2*i+1] - (REAL)x[2*i+1];
      double h = hypot(d0,d1);
      printf("  %2d: %10f,%10f   -- %10f,%10f  %s\n",(int)i,(REAL)x[2*i],(REAL)x[2*i+1],(REAL)x_ref[2*i],(REAL)x_ref[2*i+1],(h>1.0e-4)?"****":"");
    }
  // Check permutation
  printf("Permutation:");
  for (size_t i=0;i<n;i++)
  {
    int k = -1;
    for (size_t j=0;j<n;j++)
    {
      double d0 = x_ref[2*i] - x[2*j];
      double d1 = x_ref[2*i+1] - x[2*j+1];
      double h = sqrt(d0*d0+d1*d1);
      if (h<1.0e-4) { k = (int)j; break; }
    }
    printf("%c%d",(i==0)?' ':',',k);
  }
  printf("\n");
}


// Check and report OpenCL errors
// Copyright 2011, Eric Bainville
const int NOpenCLErrorCodes = 63;
static const char * OpenCLErrorCodes[NOpenCLErrorCodes] = {
  "CL_SUCCESS", // 0
  "CL_DEVICE_NOT_FOUND", // -1
  "CL_DEVICE_NOT_AVAILABLE", // -2
  "CL_COMPILER_NOT_AVAILABLE", // -3
  "CL_MEM_OBJECT_ALLOCATION_FAILURE", // -4
  "CL_OUT_OF_RESOURCES", // -5
  "CL_OUT_OF_HOST_MEMORY", // -6
  "CL_PROFILING_INFO_NOT_AVAILABLE", // -7
  "CL_MEM_COPY_OVERLAP", // -8
  "CL_IMAGE_FORMAT_MISMATCH", // -9
  "CL_IMAGE_FORMAT_NOT_SUPPORTED", // -10
  "CL_BUILD_PROGRAM_FAILURE", // -11
  "CL_MAP_FAILURE", // -12
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // -13..-29
  "CL_INVALID_VALUE", // -30
  "CL_INVALID_DEVICE_TYPE", // -31
  "CL_INVALID_PLATFORM", // -32
  "CL_INVALID_DEVICE", // -33
  "CL_INVALID_CONTEXT", // -34
  "CL_INVALID_QUEUE_PROPERTIES", // -35
  "CL_INVALID_COMMAND_QUEUE", // -36
  "CL_INVALID_HOST_PTR", // -37
  "CL_INVALID_MEM_OBJECT", // -38
  "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", // -39
  "CL_INVALID_IMAGE_SIZE", // -40
  "CL_INVALID_SAMPLER", // -41
  "CL_INVALID_BINARY", // -42
  "CL_INVALID_BUILD_OPTIONS", // -43
  "CL_INVALID_PROGRAM", // -44
  "CL_INVALID_PROGRAM_EXECUTABLE", // -45
  "CL_INVALID_KERNEL_NAME", // -46
  "CL_INVALID_KERNEL_DEFINITION", // -47
  "CL_INVALID_KERNEL", // -48
  "CL_INVALID_ARG_INDEX", // -49
  "CL_INVALID_ARG_VALUE", // -50
  "CL_INVALID_ARG_SIZE", // -51
  "CL_INVALID_KERNEL_ARGS", // -52
  "CL_INVALID_WORK_DIMENSION", // -53
  "CL_INVALID_WORK_GROUP_SIZE", // -54
  "CL_INVALID_WORK_ITEM_SIZE", // -55
  "CL_INVALID_GLOBAL_OFFSET", // -56
  "CL_INVALID_EVENT_WAIT_LIST", // -57
  "CL_INVALID_EVENT", // -58
  "CL_INVALID_OPERATION", // -59
  "CL_INVALID_GL_OBJECT", // -60
  "CL_INVALID_BUFFER_SIZE", // -61
  "CL_INVALID_MIP_LEVEL" // -62
};

bool clfft::checkStatus(int status,const char * filename,int line)
{
  if (status == 0) return true; // OK
  int e = -status;
  if (e >= 0 && e < NOpenCLErrorCodes && OpenCLErrorCodes[e] != 0)
    fprintf(stderr,"%s:%d: OpenCL error %s\n",filename,line,OpenCLErrorCodes[e]);
  else
    fprintf(stderr,"%s:%d: OpenCL error %d\n",filename,line,-status);
  abort();
  return false; // Error
}


//
// OpenCL FFT benchmarks
// Copyright 2011, Eric Bainville
// Modified by Beau Johnston Sep 2017.
// All rights reserved.
//
#include "common_args.h"
#include "rdtsc.h"

// Return log2(n) if N is a power of 2, and -1 otherwise.
inline int log2(size_t n)
{
  int k = 0;
  if (n <= 0) return -1; // Not a power of 2
  while (n != (size_t)1)
  {
    if (n&(size_t)1) return -1; // Has at least 2 bits set
    n >>= (size_t)1;
    k++;
  }
  return k;
}

clfft::Context * clfft::Context::create(cl_context context,RealType realType,std::string & errorMsg)
{
  errorMsg.clear();
  // Check args
  if (context == 0) { errorMsg.assign("Invalid OpenCL context"); return 0; } // Invalid context
  if (realType < 0 || realType >= NB_CLFFT_REAL_TYPES) { errorMsg.assign("Invalid data type"); return 0; } // Invalid type

  bool ok = true;
  cl_int status;
  size_t sz;
  int nDevices = 1;
  clfft::Context * result = 0;
  const int MAX_OPTIONS = 1024;
  char options[MAX_OPTIONS];
  
  // Alloc result
  result = new clfft::Context();
  if (result == 0) return 0; // Alloc failed
  result->mRealType = realType;
  result->mRealTypeSize = (realType == FLOAT_REAL_TYPE)?sizeof(float):sizeof(double);
  result->mContext = context;
  status = clRetainContext(context);

  if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }

  // Create and build program
   
  if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }
  snprintf(options,MAX_OPTIONS," -cl-fast-relaxed-math -D CONFIG_USE_DOUBLE=%d",
    (realType == DOUBLE_REAL_TYPE)?1:0
  );
  result->mProgram = ocdBuildProgramFromFile(context,device_id,"fft_kernels",options);
  
  // Create kernels
  result->mKernel.resize(NB_KERNELS,0);
  result->mKernelIndex.resize(NB_KERNELS,0);
  for (int i=0;i<NB_KERNELS;i++)
  {
    const char * kname = 0;
    switch (i)
    {
    case UNPACK_REAL1D_KERNEL: kname = "unpackReal1DKernel"; break;
    case UNPACK_COMPLEX1D_KERNEL: kname = "unpackComplex1DKernel"; break;
    case FFT_RADIX2_KERNEL: kname = "fftRadix2Kernel"; break;
    case FFT_RADIX4_KERNEL: kname = "fftRadix4Kernel"; break;
    case FFT_RADIX8_KERNEL: kname = "fftRadix8Kernel"; break;
    case FFT_RADIX16_KERNEL: kname = "fftRadix16Kernel"; break;
    default: break;
    }
    if (kname == 0) { ok = false; break; } // Failed
    result->mKernel[i] = clCreateKernel(result->mProgram,kname,&status);
    if (!CLFFT_CHECK_STATUS(status)) { errorMsg.append("Kernel creation failed: "); errorMsg.append(kname); ok = false; goto END; }
  }

  // Create command queues
  result->mQueue.resize(nDevices,0);
  result->mDevice.resize(nDevices,0);
  for (int d=0;d<nDevices;d++)
  {
    cl_command_queue_properties props = 0;
    result->mDevice[d] = device_id;//device[d];
    result->mQueue[d] = commands;//clCreateCommandQueue(context,device[d],props,&status);
    if (!CLFFT_CHECK_STATUS(status)) { errorMsg.append("Command queue creation failed"); ok = false; goto END; }
  }

  // Keep max device workgroup size
  result->mMaxWorkGroupSize.resize(nDevices,(size_t)0);
  for (int d=0;d<nDevices;d++)
  {
    status = clGetDeviceInfo(result->mDevice[d],CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&(result->mMaxWorkGroupSize[d]),0);
    if (!CLFFT_CHECK_STATUS(status)) { errorMsg.append("MAX_WORK_GROUP_SIZE query failed"); ok = false; goto END; }
  }

 END:
  if (!ok) { delete result; return 0; } // Error
  return result;
}

clfft::Context::Context() : mContext(0), mProgram(0)
{
}

clfft::Context::~Context()
{
  // Command queues are owned by common_args / ocd_initCL.
  for (std::vector<cl_kernel>::iterator it = mKernel.begin(); it != mKernel.end(); it++) if (*it != 0) { clReleaseKernel(*it); }
  mQueue.clear();
  mKernel.clear();
  mDevice.clear();
  if (mProgram != 0) clReleaseProgram(mProgram);
  // Context is owned by common_args / ocd_initCL.
}

int clfft::Context::getNDevices() const
{
  return (int)mDevice.size();
}

int clfft::Context::getRealType() const
{
  return mRealType;
}

size_t clfft::Context::getRealTypeSize() const
{
  return mRealTypeSize;
}

cl_context clfft::Context::getOpenCLContext() const
{
  return mContext;
}

cl_int clfft::Context::finish(int device)
{
  if (device < 0 || device >= (int)mDevice.size()) return CL_INVALID_DEVICE;
  return clFinish(mQueue[device]);
}

cl_int clfft::Context::enqueueBarrier(int device)
{
  if (device < 0 || device >= (int)mDevice.size()) return CL_INVALID_DEVICE;
  return clEnqueueBarrier(mQueue[device]);
}

cl_mem clfft::Context::createComplexBuffer(cl_mem_flags flags,size_t n,void * hostPtr)
{
  cl_int status;
  cl_mem m = clCreateBuffer(mContext,flags,2 * n * mRealTypeSize,hostPtr,&status);
  if (!CLFFT_CHECK_STATUS(status)) return 0;
  return m;
}

clfft::Event clfft::Context::enqueueRead(int device,cl_mem buffer,cl_bool blocking_read,size_t offset,size_t cb,void * ptr,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  cl_event e;
  cl_int status;
  status = clEnqueueReadBuffer(mQueue[device],buffer,blocking_read,offset,cb,ptr,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e,status);
}

clfft::Event clfft::Context::enqueueWrite(int device,cl_mem buffer,cl_bool blocking_write,size_t offset,size_t cb,const void * ptr,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  cl_event e;
  cl_int status;
  status = clEnqueueWriteBuffer(mQueue[device],buffer,blocking_write,offset,cb,ptr,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e);
}

clfft::Event clfft::Context::enqueueCopy(int device,cl_mem src,cl_mem dst,size_t src_offset,size_t dst_offset,size_t cb,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  cl_event e;
  cl_int status;
  status = clEnqueueCopyBuffer(mQueue[device],src,dst,src_offset,dst_offset,cb,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e);
}

clfft::Event clfft::Context::enqueueUnpackReal1D(int device,size_t n,size_t batch,size_t p,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  cl_kernel kernel = mKernel[UNPACK_REAL1D_KERNEL];
  int index = 0;
  cl_int status;
  cl_int pp = (cl_int)p;
  cl_event e;
  status = clSetKernelArg(kernel,index++,sizeof(cl_mem),&in); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clSetKernelArg(kernel,index++,sizeof(cl_mem),&out); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clSetKernelArg(kernel,index++,sizeof(cl_int),&pp); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clEnqueueNDRangeKernel(mQueue[device],kernel,1,0,&n,&wg,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e); // OK
}

clfft::Event clfft::Context::enqueueUnpackComplex1D(int device,size_t n,size_t batch,size_t p,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  cl_kernel kernel = mKernel[UNPACK_COMPLEX1D_KERNEL];
  int index = 0;
  cl_int status;
  cl_int pp = (cl_int)p;
  cl_event e;
  status = clSetKernelArg(kernel,index++,sizeof(cl_mem),&in); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clSetKernelArg(kernel,index++,sizeof(cl_mem),&out); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clSetKernelArg(kernel,index++,sizeof(cl_int),&pp); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clEnqueueNDRangeKernel(mQueue[device],kernel,1,0,&n,&wg,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e); // OK
}

clfft::Event clfft::Context::enqueueRadixRKernel(int device,size_t n,size_t batch,size_t p,size_t radix,Direction direction,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList)
{
  int kernelID = -1;
  switch (radix)
  {
  case 2: kernelID = FFT_RADIX2_KERNEL; break;
  case 4: kernelID = FFT_RADIX4_KERNEL; break;
  case 8: kernelID = FFT_RADIX8_KERNEL; break;
  case 16: kernelID = FFT_RADIX16_KERNEL; break;
  }
  if (kernelID < 0) return clfft::Event(CL_INVALID_ARG_VALUE);
  clearArgs(kernelID);
  pushArg(kernelID,in);
  pushArg(kernelID,out);
  pushArg<cl_int>(kernelID,(cl_int)p);
  return enqueueKernel(device,kernelID,n/radix,batch,wg,waitList);
}

clfft::Event clfft::Context::enqueueKernel(int device,int kernelID,size_t n,size_t batch,size_t wg,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  if (n <= 0 || batch <= 0) return clfft::Event(CL_INVALID_WORK_ITEM_SIZE);
  if (wg <= 0) return clfft::Event(CL_INVALID_WORK_GROUP_SIZE);
  cl_int status;

  // Limit WG if needed, by max device wg size, by max kernel wg size, by number of X threads
  wg = std::min(wg,n);
  wg = std::min(wg,mMaxWorkGroupSize[device]);
  size_t maxKWG;
  status = clGetKernelWorkGroupInfo(mKernel[kernelID],mDevice[device],CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t),&maxKWG,0);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  wg = std::min(wg,maxKWG);

  size_t wx = n; // Is a power of 2
  while ( (wx & wg) == 0 ) wx >>= 1; // remains a power of 2

  size_t nThreads[2];
  nThreads[0] = n;
  nThreads[1] = batch;
  size_t workGroup[2];
  workGroup[0] = wx;
  workGroup[1] = (size_t)1;
  cl_event e;
  status = clEnqueueNDRangeKernel(mQueue[device],mKernel[kernelID],(batch==(size_t)1)?1:2,0,nThreads,workGroup,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e); // OK
}


#if 0
cl_int CLFFTContext::enqueueFFT1D(int device,Direction direction,cl_mem in,cl_mem out,size_t n,cl_uint nEventsInWaitList,const cl_event * waitList,cl_event * e)
{
  if (e != 0) *e = 0; // Initialize return value
  int ln = log2(n);
  if (ln<0 || n<=1) return CL_INVALID_VALUE; // Invalid N

  // Buffers, 0 is IN and 1 is OUT
  cl_mem b[2] = { 0 };
  b[0] = in;
  b[1] = out;
  cl_event ev[2] = { 0 };
  cl_int status;

  // Loop on sequence size
  int current = 0; // Current buffer containing data
  size_t p = 1; // Length of combined sequences in buffer
  bool first = true;
  while (p < n)
  {
    // Select next radix of FFT to apply to the current sequences
    size_t logRadix = 1;
    if ( (p << 3) <= n ) logRadix = 3; else
    if ( (p << 2) <= n ) logRadix = 2; else
    logRadix = 1;

    // Get corresponding kernel
    int kernelID = -1;
    switch (logRadix)
    {
    case 1: kernelID = FFT_RADIX2_KERNEL; break;
    case 2: kernelID = FFT_RADIX4_KERNEL; break;
    case 3: kernelID = FFT_RADIX8_KERNEL; break;
    }
    if (kernelID < 0) { status = CL_INVALID_KERNEL; break; } // Bad log radix?

    // Enqueue kernel
    status = enqueueInOutP(device,kernelID,b[current],b[1-current],p,n >> logRadix,(first)?nEventsInWaitList:1,(first)?waitList:ev,ev+1);
    if (ev[0] != 0) clReleaseEvent(ev[0]);
    ev[0] = ev[1]; ev[1] = 0;
    if (!CLFFT_CHECK_STATUS(status)) break;

    // Prepare for next iteration
    current = 1 - current; // swap buffers
    p <<= logRadix; // increase sequence length
    first = false;
  }

  // Enqueue a final copy if the current buffer is not the output
  if (status == CL_SUCCESS && current != 1)
  {
    status = enqueueInOutP(device,UNPACK_COMPLEX1D_KERNEL,b[current],b[1-current],n,n,1,ev,ev+1);
    if (ev[0] != 0) clReleaseEvent(ev[0]);
    ev[0] = ev[1]; ev[1] = 0;
  }

  // Cleanup last event if needed
  if (status != CL_SUCCESS || e == 0)
  {
    if (ev[0] != 0) { clReleaseEvent(ev[0]); ev[0] = 0; }
  }
  // Set output event
  if (e != 0) *e = ev[0];

  return status;
}

cl_int CLFFTContext::enqueueInOutP(int device,int kernelID,cl_mem in,cl_mem out,size_t p,size_t n,cl_uint nEventsInWaitList,const cl_event * waitList,cl_event * e)
{
  if (e != 0) *e = 0; // Initialize return value
  if (device < 0 || device >= (int)mDevice.size()) return CL_INVALID_VALUE; // Invalid DEVICE
  int ln = log2(n);
  if (ln<0 || p<0 || p>n) return CL_INVALID_VALUE; // Invalid P or N

  cl_kernel kernel = mKernel[kernelID];
  cl_int status;

  // Args
  status = clSetKernelArg(kernel,0,sizeof(cl_mem),&in);
  if (!CLFFT_CHECK_STATUS(status)) return status;
  status = clSetKernelArg(kernel,1,sizeof(cl_mem),&out);
  if (!CLFFT_CHECK_STATUS(status)) return status;
  cl_int pp = (cl_int)p;
  status = clSetKernelArg(kernel,2,sizeof(cl_int),&pp);
  if (!CLFFT_CHECK_STATUS(status)) return status;

  // Group size
  size_t groupSize = n;
  size_t maxDeviceGroupSize = 0;
  size_t maxKernelGroupSize = 0;
  size_t maxGroupSize = (size_t)256; // force
  status = clGetDeviceInfo(mDevice[device],CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&maxDeviceGroupSize,0);
  if (!CLFFT_CHECK_STATUS(status)) return status;
  maxGroupSize = std::min(maxGroupSize,maxDeviceGroupSize);
  status = clGetKernelWorkGroupInfo(kernel,mDevice[device],CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t),&maxKernelGroupSize,0);
  if (!CLFFT_CHECK_STATUS(status)) return status;
  maxGroupSize = std::min(maxGroupSize,maxKernelGroupSize);
  while (groupSize > maxGroupSize) groupSize >>= (size_t)1; // keep it a divisor of N
  if (groupSize == 0) return CL_INVALID_WORK_GROUP_SIZE;

  // Enqueue
  status = clEnqueueNDRangeKernel(mQueue[device],kernel,1,0,&n,&groupSize,nEventsInWaitList,waitList,e);
  if (!CLFFT_CHECK_STATUS(status)) return status;

  return CL_SUCCESS; // OK
}
#endif


//
// OpenCL FFT TestFunctions
// Copyright 2011, Eric Bainville
// Modified by Beau Johnston. Copyright 2017 by The Australian National
// University. All rights reserved.
//

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif
#ifdef WIN32
double getRealTime()
{
  LARGE_INTEGER freq,value;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&value);
  return (double)value.QuadPart/(double)freq.QuadPart;
}
#else
double getRealTime()
{
  struct timeval tv;
  gettimeofday(&tv,0);
  return (double)tv.tv_sec + 1.0e-6*(double)tv.tv_usec;
}
#endif

// Return random double in 0..1
double rnd()
{
  double s = 0;
#if 0
  const double k = 1.0/(1.0+RAND_MAX);
  s = k * (s + (double)rand());
  s = k * (s + (double)rand());
#else
  s = rand() & 15;  // to easily test partial sums
#endif
  return s;
}

// Create an OpenCL context including all GPU devices on the first platform
// providing GPU devices.  Return a valid OpenCL context on success, and 0 otherwise.
cl_context createGPUContext()
{
  const int MAX_PLATFORMS = 8;
  const int MAX_DEVICES = 16;
  cl_platform_id platform[MAX_PLATFORMS];
  cl_device_id device[MAX_DEVICES];
  cl_uint nPlatforms = 0;
  cl_uint nDevices = 0;
  cl_context context = 0;
  cl_int status = clGetPlatformIDs(MAX_PLATFORMS,platform,&nPlatforms);
  if (status < 0 || nPlatforms == 0) return 0; // No platform

  for (cl_uint p=0;p<nPlatforms;p++)
  {
    nDevices = 0;
    status = clGetDeviceIDs(platform[p],CL_DEVICE_TYPE_GPU,MAX_DEVICES,device,&nDevices);
    if (status < 0 || nDevices == 0) continue; // Failed for this platform

    // Try to create a context using all devices
    cl_context_properties props[5];
    int index = 0;
    props[index++] = CL_CONTEXT_PLATFORM;
    props[index++] = (cl_context_properties)platform[p];
    props[index++] = 0;
    context = clCreateContext(props,nDevices,device,0,0,0);
    if (context != 0) return context; // OK
  }

  return 0; // Failed
}


//
// OpenCL FFT benchmarks
// Copyright Eric Bainville Mar 2011.
// Modified by Beau Johnston Sep 2017.
// All rights reserved.
//

#ifdef WIN32
#define USE_FFTW 1
#endif

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cmath>
#if USE_CUDA
#include <cufft.h>
#include <cuda_runtime.h>
#endif
#if USE_MKL
#include <mkl_dfti.h>
#endif
#if USE_FFTW
#include <fftw3.h>
#endif
#include "common_args.h"
#include "lsb.h"
#define BENCHMARK_IO 0

#define MIN_TIME_SEC 2

static const char* get_lsb_name()
{
    const char* name = getenv("ODW_LSB_NAME");
    return (name && name[0]) ? name : "fft";
}

static void print_fft_checksum(const char* label, const float* y, int n)
{
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;
    double checksum = 0.0;
    double abs_checksum = 0.0;

    for (int i = 0; i < 2 * n; i++) {
        double v = (double)y[i];
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (std::isinf(v)) {
            inf_count++;
            continue;
        }

        finite_count++;
        checksum += v * (double)(i + 1);
        abs_checksum += std::fabs(v);
    }

    printf("FFT_CHECKSUM label=%s signal_length=%d values=%d finite=%d nan=%d inf=%d value=%0.17e abs=%0.17e\n",
           label, n, 2 * n, finite_count, nan_count, inf_count, checksum, abs_checksum);
}


#if USE_MKL
// Run complex to complex X[2*N] to Y[2*N]. Return total time (s).
double runMKL(size_t n,const float * x,float * y,double maxBenchmarkTime)
{
  DFTI_DESCRIPTOR_HANDLE h;
  DftiCreateDescriptor(&h,DFTI_SINGLE,DFTI_COMPLEX,1,n);
  DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
  DftiCommitDescriptor(h);

  const int nops = 2;
  double t = getRealTime();
  for (int op = 0;op < nops;op++)
    {
      DftiComputeForward(h,(void *)x,y);
    }
  t = (getRealTime() - t)/(double)nops;

  DftiFreeDescriptor(&h);
  return t;
}
#endif

#if USE_FFTW
// Run complex to complex X[2*N] to Y[2*N]. Return total time (s).
double runFFTW(size_t n,const float * x,float * y,double maxBenchmarkTime)
{
  fftwf_plan p1 = fftwf_plan_dft_1d((int)n,(fftwf_complex *)x,(fftwf_complex *)y,
                                    FFTW_FORWARD,FFTW_ESTIMATE);
  double totalIT = 0;
  double t0 = getRealTime();
  double t1;
  for (int nit=1;nit<=1024;nit<<=1)
  {
    for (int it = 0;it < nit;it++)
      {
        fftwf_execute(p1);
      }
    totalIT += nit;
    t1 = getRealTime();
    if (t1 - t0 >= maxBenchmarkTime) break;
  }
  fftwf_destroy_plan(p1);
  return (t1 - t0)/totalIT;
}

bool benchmarkFFTW(size_t maxLog2N,double maxBenchmarkTime) // float only
{
  float * x = (float *) malloc((2 * sizeof(float)) << maxLog2N);
  float * y = (float *) malloc((2 * sizeof(float)) << maxLog2N);
  rand(2 << maxLog2N, x);
  for (size_t log2n = 8; log2n <= maxLog2N; log2n++)
  {
    size_t n = 1 << log2n;
    double t = runFFTW(n, x, y, maxBenchmarkTime);
    double flop = 5 * (double) log2n * (double) n;
    double perf = flop / t;
    printf("FFTW(float): N=2^%d=%d T=%.2fms  P=%.2f Gflop/s\n", (int) log2n, (int) n, t * 1.0e3, perf * 1.0e-9);
  }
  free(x);
  free(y);
  return true;
}

#endif

#if USE_CUDA
// Run complex to complex X[2*N] to Y[2*N].
double runCUFFT(size_t n,const float * x,float * y, double maxBenchmarkTime)
{
  cufftHandle plan;
  cufftComplex * inData = 0;
  cufftComplex * outData = 0;
  size_t dataSize = sizeof(cufftComplex) * n;
  cudaError_t status;
  cufftResult fftStatus;

  fftStatus = cufftPlan1d(&plan,(int)n,CUFFT_C2C,1); // 1 is BATCH size
  assert(fftStatus == CUFFT_SUCCESS);
  status = cudaMalloc((void **)(&inData),dataSize);
  assert(status == cudaSuccess);
  status = cudaMalloc((void **)(&outData),dataSize);
  assert(status == cudaSuccess);

  // Send X to device
  status = cudaMemcpy(inData,x,dataSize,cudaMemcpyHostToDevice);
  assert(status == cudaSuccess);

  double totalIT = 0;
  double t0 = getRealTime();
  double t1;
  for (int nit = 1; nit <= 1024; nit++)
  {
    for (int it = 0; it < nit; it++)
    {
      // Run the FFT
#if BENCHMARK_IO
      cudaMemcpy(inData,x,dataSize,cudaMemcpyHostToDevice);
#endif
      cufftExecC2C(plan, inData, outData, CUFFT_FORWARD);
#if BENCHMARK_IO
      cudaMemcpy(y,outData,dataSize,cudaMemcpyDeviceToHost);
#endif
    }
    cudaDeviceSynchronize();
    t1 = getRealTime();
    totalIT += nit;
    if (t1 - t0 >= maxBenchmarkTime) break;
  } // nit loop
  double t = (t1 - t0)/totalIT;

  // Get Y from device
  status = cudaMemcpy(y,outData,dataSize,cudaMemcpyDeviceToHost);
  assert(status == cudaSuccess);
  cudaDeviceSynchronize();

  cufftDestroy(plan);
  cudaFree(inData);
  cudaFree(outData);

  return t;
}

bool benchmarkCUFFT(size_t maxLog2N,double maxBenchmarkTime) // float only
{
  float * x = (float *) malloc((2 * sizeof(float)) << maxLog2N);
  float * y = (float *) malloc((2 * sizeof(float)) << maxLog2N);
  rand(2 << maxLog2N, x);
  for (size_t log2n = 8; log2n <= maxLog2N; log2n++)
  {
    size_t n = 1 << log2n;
    double t = runCUFFT(n, x, y, maxBenchmarkTime);
    double flop = 5 * (double) log2n * (double) n;
    double perf = flop / t;
    printf("CUFFT(float): N=2^%d=%d T=%.2fms  P=%.2f Gflop/s\n", (int) log2n, (int) n, t * 1.0e3, perf * 1.0e-9);
  }
  free(x);
  free(y);
  return true;
}

#endif // CUDA

clfft::Event simpleForward1D(clfft::Context * clfft,int device,size_t n,cl_mem in,cl_mem out,clfft::EventVector deps)
{
  cl_mem b[2];
  int current = 0;
  size_t p = (size_t)1;
  clfft::Event e;
  size_t bufferSize = n * 2 * clfft->getRealTypeSize();
  b[current] = in;
  b[1-current] = out;

  while (p<n)
  {
    size_t radix = (size_t)2;
    if ( (p<<4) <= n ) radix = 16;
    else if ( (p<<3) <= n ) radix = 8;
    else if ( (p<<2) <= n) radix = 4;
    else radix = 2;
    e = clfft->enqueueRadixRKernel(device,n,1,p,radix,clfft::FORWARD_DIRECTION,b[current],b[1-current],256,deps);
    if (!CLFFT_CHECK_EVENT(e)) return e;
    deps = clfft::EventVector(e);
    p *= radix;
    current = 1 - current;
  }
  if (current != 1)
  {
      e = clfft->enqueueCopy(device,b[current],b[1-current],0,0,bufferSize,deps);
  }
  return e;
}

int runFFT(clfft::Context * clfft,size_t n,void * x,void * y)
{
    int realType = clfft->getRealType();
    size_t realSize = (realType == clfft::FLOAT_REAL_TYPE)?sizeof(float):sizeof(double);
    size_t bufferSize = realSize * n * 2;
    cl_int status;
    cl_mem bIn = 0;
    cl_mem bOut = 0;
    bool ok = true;
    int deviceID = 0;
    clfft::Event e;

    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();
    bIn = clCreateBuffer(clfft->getOpenCLContext(),CL_MEM_READ_WRITE,bufferSize,0,&status);
    if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }
    bOut = clCreateBuffer(clfft->getOpenCLContext(),CL_MEM_READ_WRITE,bufferSize,0,&status);
    if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }
    LSB_Rec(0);

    LSB_Set_Rparam_string("region","device_side_h2d_copy");
    LSB_Res();
    e = clfft->enqueueWrite(deviceID,bIn,true,0,bufferSize,x,clfft::EventVector()); // blocking
    if (!CLFFT_CHECK_EVENT(e)) { ok = false; goto END; }
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "fft_kernel");
    LSB_Res();
    e = simpleForward1D(clfft,deviceID,n,bIn,bOut,e);
    if (!CLFFT_CHECK_EVENT(e)) { ok = false; goto END; }
    status = clfft->enqueueBarrier(deviceID);
    if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }
    status = clfft->finish(deviceID);
    LSB_Rec(0);

    if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }

    LSB_Set_Rparam_string("region", "device_side_d2h_copy");
    LSB_Res();
    e = clfft->enqueueRead(deviceID,bOut,true,0,bufferSize,y,e); // blocking
    if (!CLFFT_CHECK_EVENT(e)) { ok = false; goto END; }
    LSB_Rec(0);

END:

    if (bIn != 0) clReleaseMemObject(bIn);
    if (bOut != 0) clReleaseMemObject(bOut);

    if (!ok) return -1; // Error
    else return 0;
}

double runFFT(clfft::Context * clfft,size_t n,void * x,void * y, double maxBenchmarkTime)
{
  int realType = clfft->getRealType();
  size_t realSize = (realType == clfft::FLOAT_REAL_TYPE)?sizeof(float):sizeof(double);
  size_t bufferSize = realSize * n * 2;
  cl_int status;
  cl_mem bIn = 0;
  cl_mem bOut = 0;
  bool ok = true;
  double t0,t1,t,totalIT;
  int deviceID = 0;
  t = -1;
  clfft::Event e;

  bIn = clCreateBuffer(clfft->getOpenCLContext(),CL_MEM_READ_WRITE,bufferSize,0,&status);
  if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }
  bOut = clCreateBuffer(clfft->getOpenCLContext(),CL_MEM_READ_WRITE,bufferSize,0,&status);
  if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }

  e = clfft->enqueueWrite(deviceID,bIn,true,0,bufferSize,x,clfft::EventVector()); // blocking
  if (!CLFFT_CHECK_EVENT(e)) { ok = false; goto END; }

  t0 = getRealTime();
  t1 = 0;
  totalIT = 0;
  for (int nit = 1; nit <= 1024; nit <<= 1)
  {
    for (int it = 0; it < nit; it++)
    {
      e = simpleForward1D(clfft,deviceID,n,bIn,bOut,e);
      if (!CLFFT_CHECK_EVENT(e)) { ok = false; goto END; }
      status = clfft->enqueueBarrier(deviceID);
      if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }
    }
    status = clfft->finish(deviceID);
    if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }

    totalIT += nit;
    t1 = getRealTime();
    if (t1 - t0 >= maxBenchmarkTime) break; // Run 3s max test
  } // nit loop
  t = (t1 - t0) / totalIT; // s per FFT

  e = clfft->enqueueRead(deviceID,bOut,true,0,bufferSize,y,e); // blocking
  if (!CLFFT_CHECK_EVENT(e)) { ok = false; goto END; }

END:

  if (bIn != 0) clReleaseMemObject(bIn);
  if (bOut != 0) clReleaseMemObject(bOut);

  if (!ok) return -1; // Error
  return t;
}

bool benchmarkFFT(size_t maxLog2N,clfft::RealType realType,double maxBenchmarkTime)
{
  std::string msg;
  clfft::Context * clfft = 0;
  //cl_context context = 0;
  bool ok = true;
  size_t realSize;
  size_t maxBufferSize;
  size_t maxN;
  void * x = 0;
  void * y = 0;

  ocd_initCL();
  //context = createGPUContext();
  if (context == 0)
  {
    fprintf(stderr,"Could not create OpenCL context\n");
    ok = false; goto END;
  }
  clfft = clfft::Context::create(context,realType,msg);
  if (clfft == 0)
  {
    fprintf(stderr,"Creation failed:\n%s\n",msg.c_str());
    ok = false; goto END;
  }
  //clReleaseContext(context); // clfft still references the context

  realSize = (realType == clfft::FLOAT_REAL_TYPE)?sizeof(float):sizeof(double);
  maxN = (size_t)1 << maxLog2N;
  maxBufferSize = realSize * (size_t)2 * maxN;
  x = malloc(maxBufferSize);
  y = malloc(maxBufferSize);
  if (realType == clfft::FLOAT_REAL_TYPE) rand<float>((size_t)2*maxN,(float *)x);
  else rand<double>(2*maxN,(double *)x);

  for (size_t log2n=8;log2n <= maxLog2N;log2n++)
  {
    rand<float>((size_t)2*maxN,(float *)x);
    size_t n = (size_t)1 << log2n;
    double t = runFFT(clfft,n,x,y,maxBenchmarkTime);
    double flop = 5 * (double)log2n * (double)n;
    double perf = flop / t; // flop/s per FFT
    printf("FFT(%s): N=2^%d=%d T=%.2fms  P=%.2f Gflop/s\n",(realType==clfft::FLOAT_REAL_TYPE)?"float":"double",
                (int)log2n,(int)n,t*1.0e3,perf*1.0e-9);
  } // log2n loop

END:
  if (x != 0) free(x);
  if (y != 0) free(y);
  delete clfft;
  return ok;
}

bool runBenchmarks(double maxBenchmarkTime)
{
  bool ok = true;

  printf("Running benchmarks...\n");

  // BENCHMARKS
  const size_t maxLog2N = 24;

  ok &= benchmarkFFT(maxLog2N,clfft::FLOAT_REAL_TYPE,maxBenchmarkTime);
  ok &= benchmarkFFT(maxLog2N,clfft::DOUBLE_REAL_TYPE,maxBenchmarkTime);
#if USE_CUDA
  ok &= benchmarkCUFFT(maxLog2N,maxBenchmarkTime); // float only
#endif
#if USE_FFTW
  ok &= benchmarkFFTW(maxLog2N,maxBenchmarkTime);
#endif

  return ok;
}

void printHelp(){
    printf("fft performs a one dimensional fast fourier transform over a given signal length N.\n");
    printf("Arguments are supported in the following form:\n");
    printf("\t./fft-opencl -p [platform id] -d [device id] -t [type id] -- [N]\n");
    printf("\twhere: [platform id] is the integer id for the OpenCL platform to use,\n");
    printf("\t       [device id] is the integer id for the OpenCL device,\n");
    printf("\t       [type id] is the integer id for the OpenCL platform to use, by default this determines type automatically according to the selected device characteristics,\n");
    printf("\t       [N] is an integer (and must be a power of 2) an indicates the length of signal on which to perform the FFT.\n");
    printf("Additionally -d or --default, runs the default (original Gflops benchmark)\n");
    printf("sample usage:\n");
    printf("\t./fft-opencl -p 0 -d 0 -t 0 -- 128\n");
}

bool isPowerOfTwo(unsigned int x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}

int benchmark(int N){
    LSB_Init(get_lsb_name(), 0);
    LSB_Set_Rparam_int("repeats_to_two_seconds", 0);
    LSB_Set_Rparam_int("signal_length",N);

    LSB_Set_Rparam_string("region", "runtime_initialization");
    LSB_Res();
    ocd_initCL();
    LSB_Rec(0);
    std::string msg;
    LSB_Set_Rparam_string("region", "host_side_setup");
    LSB_Res();
    clfft::Context* clfft = clfft::Context::create(context,clfft::FLOAT_REAL_TYPE,msg);
    //generate the data of length N
    void* x = malloc(sizeof(float)*N*2);//    new float[N*2];
    void* y = malloc(sizeof(float)*N*2);//    new float[N*2];
    //ones<float>((size_t)N,(float*)x);
    sine<float>((size_t)N,(float*)x);
    zeros<float>((size_t)N,(float*)y);
    printf("Working kernel memory: %fKiB\n",(sizeof(float)*N*2*2)/1024.0);
    LSB_Rec(0);

    int return_code = 0;
    int lsb_timing_repeats = 0;
    struct timeval startTime, currentTime, elapsedTime;
    gettimeofday(&startTime, NULL);
    do {
        LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

        return_code &= runFFT(clfft,N,(void*)x,(void*)y);

        lsb_timing_repeats++;
        gettimeofday(&currentTime, NULL);
        timersub(&currentTime, &startTime, &elapsedTime);
    } while (elapsedTime.tv_sec < MIN_TIME_SEC);

    print_fft_checksum("fft", (const float*)y, N);

    free(x);
    free(y);

    LSB_Set_Rparam_string("region", "runtime_finalization");
    LSB_Res();
    clFinish(commands);
    LSB_Rec(0);

    LSB_Finalize();
    return(return_code);
}

int main(int argc, char**argv)
{
    bool ok = true;
    srand(0);
    ocd_init(&argc, &argv, NULL);
    if (argc == 2){
        if (strcmp(argv[1],"-h") == 0 || strcmp(argv[1],"--help") == 0){
            printHelp();
            return(EXIT_SUCCESS);
        }
        if (strcmp(argv[1],"-d") == 0 || strcmp(argv[1],"--default") == 0){
            ok &= runBenchmarks(0.5);
        }
        int signal_length = atoi(argv[1]);
        if(signal_length == 0){
            printf("Please enter a valid signal length of N elements\n");
            printHelp();
            return(EXIT_FAILURE);
        }
        if(!isPowerOfTwo(signal_length)){
            printf("N must be a power of 2, but instead is %i\n",signal_length);
            printHelp();
            return(EXIT_FAILURE);
        }
        return(benchmark(signal_length));
    }else{
        printHelp();
        return(EXIT_SUCCESS);
    }
    printf("%s\n",(ok)?"OK":"FAILED!");
#ifdef WIN32
    printf("Press a key.\n");
    getchar();
#endif
}

