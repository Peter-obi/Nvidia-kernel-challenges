#define TILE_WIDTH 16

__global__ void vecMatMul(const float* __restrict__ M,
                          const float* __restrict__ x,
                          float* __restrict__ P,
                          int m, int k)
{
    extern __shared__ float xds[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    int full_tiles = k / TILE_WIDTH;
    int tail = k % TILE_WIDTH;
    
    for (int ph = 0; ph < full_tiles; ++ph) {
        int kbase = ph * TILE_WIDTH;
        if (threadIdx.x < TILE_WIDTH)
            xds[threadIdx.x] = x[kbase + threadIdx.x];
        __syncthreads();
        
        if (row < m) {
            #pragma unroll
            for (int i = 0; i < TILE_WIDTH; ++i){
            sum += M[row * k + kbase + i] * xds[i];
            }
        }
        __syncthreads();
    }
    
    if (tail > 0) {
        int kbase = full_tiles * TILE_WIDTH;
        if (threadIdx.x < TILE_WIDTH)
            xds[threadIdx.x] = (threadIdx.x < tail) ? x[kbase + threadIdx.x] : 0.0f;
        __syncthreads();
        
        if (row < m) {
            for (int i = 0; i < tail; ++i)
                sum += M[row * k + kbase + i] * xds[i];
        }
        __syncthreads();
    }
    
    if (row < m)
        P[row] = sum;
}
