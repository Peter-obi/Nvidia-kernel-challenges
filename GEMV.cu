#define TILE 128   

//A is a matrix m x n
//x is vector of length n
//y is length m

__global__
void gemv_tiled_x(const float* __restrict__ A,
                  const float* __restrict__ x,
                  float* __restrict__ y,
                  int m, int n)
{
    extern __shared__ float xds[];   

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    float acc = 0.0f;

    int num_tiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t)
    {
        int base = t * TILE;

        for (int i = threadIdx.x; i < TILE; i += blockDim.x)
        {
            int k = base + i;
            xds[i] = (k < n) ? x[k] : 0.0f;
        }

        __syncthreads();

        // Number of valid elements in this tile (handles last partial tile)
        int limit = min(TILE, n - base);

        // Dot product contribution for this tile
        const float* Arow = &A[row * n + base];
        for (int k = 0; k < limit; ++k)
        {
            acc += Arow[k] * xds[k];
        }

        __syncthreads();
    }

    y[row] = acc;
}
