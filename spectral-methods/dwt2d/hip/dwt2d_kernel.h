#define THREADS 256
#define BOUNDARY_X 2


__device__ int divRndUp(int n, 
             int d)
{
    return (n / d) + ((n % d) ? 1 : 0);
}


/* Store 3 RGB float components */
/*void storeComponents(float *d_r, float *d_g, float *d_b, const float r, const float g, const float b, int pos) 
{
    d_r[pos] = (r/255.0f) - 0.5f;
    d_g[pos] = (g/255.0f) - 0.5f;
    d_b[pos] = (b/255.0f) - 0.5f;
}
*/


/* Store 3 RGB intege components */
/*void storeComponents(int *d_r,   
                     int *d_g,
                     int *d_b,
                     int r,
                     int g,
                     int b,
                     int pos)
{
    d_r[pos] = r - 128;
    d_g[pos] = g - 128;
    d_b[pos] = b - 128;
}   
*/

/* Store float component */
/*__global__ void storeComponent(float *d_c, const float c, int pos)
{
    d_c[pos] = (c/255.0f) - 0.5f;
}
*/


// Store integer component 
__device__ void storeComponent(int *d_c,
                    const int c,
                    int pos)
{
    d_c[pos] = c - 128;
}


// Copy img src data into three separated component buffers, and convert from
// unsigned char* to signed int* positioned around 0
__global__ void c_CopySrcToComponents (int *d_r,
                                     int *d_g,
                                     int *d_b,
                                     unsigned char * cl_d_src,
                                     int pixels)
{
    int x = threadIdx.x;
    int gX= blockDim.x * blockIdx.x; 
    int i = gX + x;

    if(i < pixels){
        d_r[i] = (int)cl_d_src[i*3+0] - 128;
        d_g[i] = (int)cl_d_src[i*3+1] - 128;
        d_b[i] = (int)cl_d_src[i*3+2] - 128;
    }
}


// Copy img src data into three separated component buffers 
__global__ void c_CopySrcToComponent (int *d_c, 
									unsigned char * cl_d_src,
									int pixels)
{
	int x = threadIdx.x;
	int gX = blockDim.x * blockIdx.x;
	
	__shared__ unsigned char sData[THREADS];
	
	sData[ x ] = cl_d_src [gX + x];
	
	__syncthreads();
	
	int c;
	
	c = (int) (sData[x]);
	
	int globalOutputPosition = gX + x;
	if (globalOutputPosition < pixels)
	{
		storeComponent(d_c, c, globalOutputPosition);
	}
	
}


__device__ static void mirror( int *d,
                    const int sizeD)
{
	if ((*d )>= sizeD)
	{
		(*d) = 2 * sizeD -2 - (*d);
	} else if((*d) < 0)
	{
		(*d) = -(*d) ;
	}
}


struct VerticalDWTPixelIO
{
	bool CHECKED;
	int end, stride;
};


__device__ int initialize_PixelIO(struct VerticalDWTPixelIO *pIO,
                       bool CHECK,
                       const int sizeX,
                       const int sizeY,
                       int firstX,
                       int firstY)
{
	pIO->CHECKED = CHECK;
	pIO->end = pIO->CHECKED ? (sizeY * sizeX + firstX) : 0 ;
	pIO->stride = sizeX;
    return firstX + sizeX * firstY;
}


struct VerticalDWTPixelLoader
{
	bool CHECKED;
	int last;
};


__device__ void init_PixelLoader(struct VerticalDWTPixelLoader *loader,
                      const int sizeX,
                      const int sizeY,
                      int firstX,
                      const int firstY,
                      struct VerticalDWTPixelIO *pIO,
                      bool CHECK )
{
	mirror (&firstX, sizeX);
	loader->last = initialize_PixelIO (pIO, CHECK, sizeX, sizeY, firstX, firstY) - sizeX;
}


__device__ void clear_PixelLoader(struct VerticalDWTPixelLoader *pLoader,
                       struct VerticalDWTPixelIO *pIO)
{
	pLoader->last = 0;
	pIO->end = 0 ;
	pIO->stride = 0 ;
}


__device__ int loadFrom(struct VerticalDWTPixelLoader *pLoader,
             const int * const input,
             struct VerticalDWTPixelIO *pIO,
             int CHECK)
{	
	pLoader->last += pIO->stride;   
	if(CHECK && (pLoader->last == pIO->end)) 
	{
        pLoader->last -= 2 * pIO->stride;
        pIO->stride = 0 - pIO->stride;  
    }
	return input[pLoader->last];  
} 


struct VerticalDWTBandIO 
{
	bool CHECKED;
	int end;
	int strideHighToLow;
	int strideLowToHigh;
};


__device__ int initialize_BandIO(struct VerticalDWTBandIO *bandIO,
                      const int imageSizeX,
                      const int imageSizeY,
                      int firstX,
                      int firstY) 
{
	int columnOffset = firstX / 2;
    int verticalStride;
	
	if(firstX & 1)
	{
		verticalStride = imageSizeX / 2;
		columnOffset += divRndUp(imageSizeX, 2) * divRndUp(imageSizeY, 2);
        bandIO->strideLowToHigh = (imageSizeX * imageSizeY) / 2;
	}
	else
	{
	verticalStride = imageSizeX / 2 + (imageSizeX & 1);
    bandIO->strideLowToHigh = divRndUp(imageSizeY, 2)  * imageSizeX;		
	} 
	
	bandIO->strideHighToLow = verticalStride - bandIO->strideLowToHigh;
	
	if (bandIO->CHECKED) 
	{
		bandIO->end = columnOffset + (imageSizeY / 2) * verticalStride + (imageSizeY & 1) * bandIO->strideLowToHigh ;
	}
	else
	{
		bandIO->end = 0;
	}
	
	return columnOffset + (firstY / 2) * verticalStride    // right row
              + (firstY & 1) * bandIO->strideLowToHigh;
} 


struct VerticalDWTBandLoader
{
	bool CHECKED;
	int last;
};


struct VerticalDWTBandWriter
{
	bool CHECKED;  
	int next;
};


__device__ int saveAndUpdate(struct VerticalDWTBandWriter *writer,
                  bool CHECK,
                  struct VerticalDWTBandIO *bandIO,
                  int * const output,
                  int *item,
                  int *stride) 
{
	writer->CHECKED = CHECK;   
	if((!writer->CHECKED) || (writer->next != bandIO->end) )  
	{
		output[writer->next] = *item;
        writer->next += *stride;
    } 
	return writer->next; 
}


__device__ void clear_BandWriter(struct VerticalDWTBandWriter *writer,
                      struct VerticalDWTBandIO *bandIO)
{
	bandIO->end = 0;
	bandIO->strideHighToLow = 0;
	bandIO->strideLowToHigh = 0;
	writer->next = 0 ;
}


__device__ void init_BandWriter(struct VerticalDWTBandWriter *writer,
                     struct VerticalDWTBandIO *bandIO,
                     const int imageSizeX,
                     const int imageSizeY,
                     const int firstX,
                     const int firstY)
{
	if (firstX < imageSizeX)
	{ 
		writer->next = initialize_BandIO (bandIO, imageSizeX, imageSizeY, firstX, firstY);
	}
	else
	{
		clear_BandWriter (writer , bandIO) ;
	}
}


__device__ int writeLowInto(struct VerticalDWTBandWriter *writer,
                 struct VerticalDWTBandIO *bandIO,
                 int * const output,
                 int *primary)
{
	return saveAndUpdate(writer, writer->CHECKED, bandIO, output, primary, &(bandIO->strideLowToHigh));
}

__device__ int writeHighInto(struct VerticalDWTBandWriter *writer, struct VerticalDWTBandIO *bandIO,  int * const output, int *other)
{
	return saveAndUpdate(writer, writer->CHECKED, bandIO, output, other,   &(bandIO->strideHighToLow));
}


//TransformBuffer is contained in cuda_gwt/transform_buffer.h
struct TransformBuffer
{
	int SIZE_X, SIZE_Y;  
	int VERTICAL_STRIDE; 
	int SHM_BANKS, BUFFER_SIZE, PADDING, ODD_OFFSET;
	
	/// buffer for both even and odd columns
    int *data;     //data[2 * BUFFER_SIZE + PADDING]
};


__device__ void horizontalStep(struct TransformBuffer *buffer,
                     const int count,
                     const int prevOffset,
                     const int midOffset,
                     const int nextOffset,
                     int flag)
{
	const int STEPS = count / buffer->SIZE_X;
	const int finalCount = count % buffer->SIZE_X; 
    const int finalOffset = count - finalCount; 
	for(int i = 0; i< STEPS; i++)
	{
		const int previous = buffer->data[prevOffset + i * buffer->SIZE_X + threadIdx.x] ;
		const int next     = buffer->data[nextOffset + i * buffer->SIZE_X + threadIdx.x];
		int * center = & (buffer->data[midOffset + i *  buffer->SIZE_X + threadIdx.x]); 
		if (flag == 0)
		{
			*center -= (previous + next) /2; //Forward53Predict()
		} else if (flag == 1)
		{
			*center += (previous + next + 2) /4; //Forward53Update()
		}
	}
	
	if(threadIdx.x < finalCount) {
        const int previous = buffer->data[prevOffset + finalOffset + threadIdx.x];
        const int next     = buffer->data[nextOffset + finalOffset + threadIdx.x];
        int * center = & (buffer->data[midOffset + finalOffset + threadIdx.x]);
		
        if (flag == 0)
		{
			*center -= (previous + next) /2; //Forward53Predict() 
		} else if (flag == 1)
		{
			*center += (previous + next + 2) /4; //Forward53Update()
		}
    }
	
}


__device__ void forEachHorizontalOdd(struct TransformBuffer *buffer,
                          const int firstLine,
                          const int numLines,
                          int flag) 
{
	const int count = numLines * buffer->VERTICAL_STRIDE - 1 ;
	const int prevOffset = firstLine * buffer->VERTICAL_STRIDE ; 
	const int centerOffset = prevOffset + buffer->ODD_OFFSET ; 
	const int nextOffset = prevOffset + 1;
	
	horizontalStep (buffer, count, prevOffset, centerOffset, nextOffset, flag);

}


__device__ void forEachHorizontalEven(struct TransformBuffer *buffer,
                           const int firstLine,
                           const int numLines,
                           int flag) 
{
	const int count = numLines * buffer->VERTICAL_STRIDE - 1 ;
	const int centerOffset = firstLine * buffer->VERTICAL_STRIDE + 1; 
	const int prevOffset = firstLine * buffer->VERTICAL_STRIDE + buffer->ODD_OFFSET; 
	const int nextOffset = prevOffset + 1;
	
	horizontalStep (buffer, count, prevOffset, centerOffset, nextOffset, flag);
}


__device__ void forEachVerticalOdd(struct TransformBuffer *buffer,
                         const int columnOffset,
                         int flag)
{
	int steps = (buffer->SIZE_Y - 1) / 2;
	for (int i = 0; i < steps; i++)
	{
		int row = i * 2 + 1;
		int prev = buffer->data[columnOffset+ (row - 1) * buffer->VERTICAL_STRIDE];
		int next = buffer->data[columnOffset+ (row + 1) * buffer->VERTICAL_STRIDE];

		if (flag == 0)
		{
			buffer->data[columnOffset + row * buffer->VERTICAL_STRIDE] -= (prev + next) /2;	 	
		}
		else if (flag == 1)
		{
			//buffer->data[columnOffset + row * buffer->VERTICAL_STRIDE] += (prev + next + 2) /4;
		}
	}
}


__device__ void forEachVerticalEven(struct TransformBuffer *buffer,
                          const int columnOffset,
                          int flag)
{
	int i ;
	if(buffer->SIZE_Y > 3)
	{ 
		int steps = (int)( buffer->SIZE_Y / 2) -1 ;
		
		for(i = 0; i < steps; i++) 
		{
			int row = 2 + i * 2;
			int prev = buffer->data[columnOffset+ (row - 1) * buffer->VERTICAL_STRIDE];
			int next = buffer->data[columnOffset + (row + 1) * buffer->VERTICAL_STRIDE];
			
			if (flag == 0)
			{
				//buffer->data[columnOffset + row * buffer->VERTICAL_STRIDE] -= (prev + next) /2; 
			}
			else if (flag == 1)
			{
				buffer->data[columnOffset + row * buffer->VERTICAL_STRIDE] += (prev + next + 2)/4; //real one
			}
			
        }
	}	
}


struct FDWT53Column
{
	bool CHECKED_LOADER;
	// loader for the column
	struct VerticalDWTPixelLoader loader;
	/// offset of the column in shared buffer
    int offset;                   
    // backup of first 3 loaded pixels (not transformed)
    int pixel0, pixel1, pixel2;

};


__device__ void clear_FDWT53Column(struct FDWT53Column *st_FDWT53Column,
                        struct VerticalDWTPixelIO *pIO)
{
	st_FDWT53Column->offset = 0;
	st_FDWT53Column->pixel0 = 0;
	st_FDWT53Column->pixel1 = 0;
	st_FDWT53Column->pixel2 = 0;
	clear_PixelLoader(&(st_FDWT53Column->loader), pIO);
}


struct FDWT53 {
	int WIN_SIZE_X, WIN_SIZE_Y;
	struct FDWT53Column column;
	/// Type of shared memory buffer for 5/3 FDWT transforms.
	/// Actual shared buffer used for forward 5/3 DWT.
    struct TransformBuffer buffer;
	
	/// Difference between indices of two vertical neighbors in buffer.
	int STRIDE;
};


//in from transform_buffer.h  
__device__ int getColumnOffset(int columnIndex,
                    struct TransformBuffer * buffer) 
{
	columnIndex += BOUNDARY_X;  
	return columnIndex / 2        // select right column
          + (columnIndex & 1) * buffer->ODD_OFFSET;  // select odd or even buffer         
}


__device__ void initColumn(struct FDWT53 * fdwt53,
                struct FDWT53Column *column,
                bool CHECKED,
                const int * const input,
                const int sizeX,
                const int sizeY,
                const int colIndex,
                const int firstY,
                struct VerticalDWTPixelIO *pIO)
{	
	column->CHECKED_LOADER = CHECKED;
	column->offset = getColumnOffset(colIndex, &fdwt53->buffer);
	
	const int firstX = blockIdx.x * fdwt53->WIN_SIZE_X + colIndex;
	if(blockIdx.y == 0) 
	{
        // topmost block - apply mirroring rules when loading first 3 rows
		init_PixelLoader(&(column->loader), sizeX, sizeY, firstX, firstY, pIO, CHECKED);
		column->pixel2 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #0
        column->pixel1 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #1
        column->pixel0 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #2
		init_PixelLoader(&(column->loader), sizeX, sizeY, firstX, firstY + 1, pIO, CHECKED);
	} 
	else
	{
		init_PixelLoader(&(column->loader), sizeX, sizeY, firstX, firstY - 2, pIO, CHECKED);
		column->pixel0 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #0
        column->pixel1 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #1
        column->pixel2 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #2
	}
	
}


__device__ void loadAndVerticallyTransform(struct FDWT53 *fdwt53,
                                 struct FDWT53Column *column,
                                 bool CHECKED,
                                 const int * const input,
                                 struct VerticalDWTPixelIO *pIO)
{
    fdwt53->buffer.data[column->offset + 0 * fdwt53->STRIDE] = column->pixel0;
    fdwt53->buffer.data[column->offset + 1 * fdwt53->STRIDE] = column->pixel1;
    fdwt53->buffer.data[column->offset + 2 * fdwt53->STRIDE] = column->pixel2;
    for (int i = 3; i < (3 + fdwt53->WIN_SIZE_Y); i++) 
    {
        fdwt53->buffer.data[column->offset + i * fdwt53->STRIDE] = loadFrom(&(column->loader),input, pIO, CHECKED);
    } 

	column->pixel0 = fdwt53->buffer.data [column->offset + ( fdwt53->WIN_SIZE_Y + 0 ) * fdwt53->STRIDE] ;
	column->pixel1 = fdwt53->buffer.data [column->offset + ( fdwt53->WIN_SIZE_Y + 1 ) * fdwt53->STRIDE] ;
	column->pixel2 = fdwt53->buffer.data [column->offset + ( fdwt53->WIN_SIZE_Y + 2 ) * fdwt53->STRIDE] ;
	
	
	int flag = 0 ;
	forEachVerticalOdd (&fdwt53->buffer, column->offset, flag);
	flag = 1 ;
	forEachVerticalEven(&fdwt53->buffer, column->offset, flag);

}


__device__ void transform(struct FDWT53 *fdwt53,
               bool CHECK_LOADS,
               bool CHECK_WRITES,
               const int * const in,
               int * out,
               const int sizeX,
               const int sizeY,
               const int winSteps)
{ 		
	// info about one main and one boundary columns processed by this thread
	struct FDWT53Column column; column.CHECKED_LOADER = CHECK_LOADS; 
    struct VerticalDWTPixelIO pIO;
    struct FDWT53Column boundaryColumn; boundaryColumn.CHECKED_LOADER = CHECK_LOADS; 
    struct VerticalDWTPixelIO pIO_b;

	// Initialize all column info: initialize loaders, compute offset of 
    // column in shared buffer and initialize loader of column.
	const int firstY = blockIdx.y * fdwt53->WIN_SIZE_Y * winSteps;
	initColumn(fdwt53, &column, CHECK_LOADS, in, sizeX, sizeY, threadIdx.x, firstY, &pIO); 

	
	// first 3 threads initialize boundary columns, others do not use them
	clear_FDWT53Column(&boundaryColumn, &pIO_b);
	if (threadIdx.x < 3) {
	// index of boundary column (relative x-axis coordinate of the column)
	const int colId = threadIdx.x + ((threadIdx.x== 0) ? fdwt53->WIN_SIZE_X : -3);
		
	// initialize the column		
	initColumn (fdwt53, &boundaryColumn, CHECK_LOADS, in, sizeX, sizeY, colId, firstY, &pIO_b);
	}
	
	// index of column which will be written into output by this thread
	const int outColumnIndex = (threadIdx.x * 2) - (fdwt53->WIN_SIZE_X - 1) * (threadIdx.x / ( fdwt53->WIN_SIZE_X / 2));
	
	// offset of column which will be written by this thread into output
    const int outColumnOffset = getColumnOffset(outColumnIndex, &(fdwt53->buffer));
	  
	// initialize output writer for this thread
    const int outputFirstX = blockIdx.x * fdwt53->WIN_SIZE_X +outColumnIndex;  
	struct VerticalDWTBandWriter writer;  writer.CHECKED = CHECK_WRITES;
	struct VerticalDWTBandIO bandIO; bandIO.CHECKED = CHECK_WRITES;
	
	init_BandWriter(&writer, &bandIO, sizeX, sizeY, outputFirstX, firstY);

	 
	// Sliding window iterations:
    // Each iteration assumes that first 3 pixels of each column are loaded.

	for(int w = 0; w < winSteps; w++)
	{

		loadAndVerticallyTransform(fdwt53, &column, CHECK_LOADS, in, &pIO); 
		if (threadIdx.x < 3)
		{
			loadAndVerticallyTransform(fdwt53, &boundaryColumn, CHECK_LOADS, in, &pIO_b); 
		}
	
		__syncthreads(); 

		int flag = 0; //flag = 0 execute Forward53Predict, flag = 1 execute Forward53Update

		forEachHorizontalOdd(&(fdwt53->buffer), 2, fdwt53->WIN_SIZE_Y, flag);		
		__syncthreads();

		flag = 1;
		forEachHorizontalEven(&(fdwt53->buffer), 2, fdwt53->WIN_SIZE_Y, flag);		
 		__syncthreads();	

		
		for(int r = 2; r < (2+fdwt53->WIN_SIZE_Y); r+= 2)
		{	
			writeLowInto(&writer,  &bandIO, out, &(fdwt53->buffer.data[outColumnOffset + r * fdwt53->buffer.VERTICAL_STRIDE]));
			writeHighInto(&writer, &bandIO, out, &(fdwt53->buffer.data[outColumnOffset + (r+1) * fdwt53->buffer.VERTICAL_STRIDE]));
		}
		
		__syncthreads();
	}

}


// Forward 5/3 DWT predict operation. 
__device__ void Forward53Predict(const int p,
                       int * c,
                       const int n) 
{
		*c -= (p + n) /2;
}
// Forward 5/3 DWT update operation.
__device__ void Forward53Update(const int p,
                      int * c,
                      const int n) 
{
		*c += (p + n + 2) /4;
}

__global__ void cl_fdwt53Kernel(const int * const in,
                              int * out,
                              const int sx,
                              const int sy,
                              const int steps,
                              int WIN_SIZE_X,
                              int WIN_SIZE_Y)
{
    extern __shared__ int shared_buffer[];
	__shared__ struct FDWT53 fdwt53;

	//initialize the TransformBuffer object
    if(threadIdx.x == 0){
        fdwt53.buffer.data = shared_buffer;
        fdwt53.WIN_SIZE_X = WIN_SIZE_X;
        fdwt53.WIN_SIZE_Y = WIN_SIZE_Y;
        fdwt53.buffer.SIZE_X = fdwt53.WIN_SIZE_X;
        fdwt53.buffer.SIZE_Y = fdwt53.WIN_SIZE_Y + 3;
        fdwt53.buffer.VERTICAL_STRIDE = BOUNDARY_X + (fdwt53.buffer.SIZE_X / 2);//BOUNDARY = 2  
        fdwt53.buffer.SHM_BANKS = 32;  // SHM_BANKS = ((__CUDA_ARCH__ >= 200) ? 32 : 16)
        fdwt53.buffer.BUFFER_SIZE = fdwt53.buffer.VERTICAL_STRIDE * fdwt53.buffer.SIZE_Y;
        fdwt53.buffer.PADDING = fdwt53.buffer.SHM_BANKS - ((fdwt53.buffer.BUFFER_SIZE + fdwt53.buffer.SHM_BANKS / 2) % fdwt53.buffer.SHM_BANKS) ;
        fdwt53.buffer.ODD_OFFSET = fdwt53.buffer.BUFFER_SIZE + fdwt53.buffer.PADDING ;
        fdwt53.STRIDE = fdwt53.buffer.VERTICAL_STRIDE ; 

        for(int i = 0; i < fdwt53.buffer.BUFFER_SIZE; i++){
            fdwt53.buffer.data[i] = 0;
        }
    }
    __syncthreads();	

    const int maxX = (blockIdx.x + 1) * WIN_SIZE_X + 1;
    const int maxY = (blockIdx.y + 1) * WIN_SIZE_Y * steps + 1;
    const bool atRightBoudary = maxX >= sx;
    const bool atBottomBoudary = maxY >= sy;
	
    // Select specialized version of code according to distance of this
    // threadblock's pixels from image boundary.
	if(atBottomBoudary)
	{
        // near bottom boundary => check both writing and reading
		transform(&fdwt53, true, true, in, out, sx, sy, steps);
	}
	else if(atRightBoudary)
	{
        // near right boundary only => check writing only
		transform(&fdwt53, false, true, in, out, sx, sy, steps);
	}
	else 
	{
        // no nearby boundary => check nothing
		transform(&fdwt53, false, false, in, out, sx, sy, steps);
	}
}
