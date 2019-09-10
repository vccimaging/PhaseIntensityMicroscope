#include "common.h"

// replicate boundary condition (for stencil size of 3x3 or 5x5)
template<int stencil_size>
__device__
void set_bd_shared_memory(volatile float (*smem)[BLOCK_Y+stencil_size-1], const float *d_in, 
                          const int tx, const int ty, const int ix, const int iy, int w, int h)
{
    const int py = iy*w;
    const int x_min = max(ix-1, 0);
    const int x_max = min(ix+1, w-1);
    const int y_min = max(iy-1, 0)  *w;
    const int y_max = min(iy+1, h-1)*w;
    switch (stencil_size)
    {
        case 3:
        {
            if (tx == 1)
            {
                smem[0][ty] = d_in[x_min + py];
                if (ty == 1)
                {
                    smem[0][0]         = d_in[x_min + y_min];
                    smem[0][BLOCK_Y+1] = d_in[x_min + y_max];
                }
            }
            if (tx == BLOCK_X)
            {
                smem[BLOCK_X+1][ty] = d_in[x_max + py];
                if (ty == 1)
                {
                    smem[BLOCK_X+1][0]         = d_in[x_max + y_min];
                    smem[BLOCK_X+1][BLOCK_Y+1] = d_in[x_max + y_max];
                }
            }
            if (ty == 1)
            {
                smem[tx][0] = d_in[ix + y_min];
            }
            if (ty == BLOCK_Y)
            {
                smem[tx][BLOCK_Y+1] = d_in[ix + y_max];
            }
        }
        break;

        case 5:
        {
            // boundary guards
            const int x_min2 = max(ix-2, 0);
            const int x_max2 = min(ix+2, w-1);
            const int y_min2 = max(iy-2, 0)  *w;
            const int y_max2 = min(iy+2, h-1)*w;

            if (tx == 2)
            {
                smem[0][ty] = d_in[x_min2 + py];
                smem[1][ty] = d_in[x_min  + py];
                if (ty == 2)
                {
                    smem[0][0] = d_in[x_min2 + y_min2];
                    smem[1][0] = d_in[x_min  + y_min2];
                    smem[0][1] = d_in[x_min2 + y_min ];
                    smem[1][1] = d_in[x_min  + y_min ];
                }
            }
            if (tx == BLOCK_X+1)
            {
                smem[BLOCK_X+2][ty] = d_in[x_max  + py];
                smem[BLOCK_X+3][ty] = d_in[x_max2 + py];
                if (ty == BLOCK_Y+1)
                {
                    smem[BLOCK_X+2][BLOCK_Y+2] = d_in[x_max  + y_max ];
                    smem[BLOCK_X+3][BLOCK_Y+2] = d_in[x_max2 + y_max ];
                    smem[BLOCK_X+2][BLOCK_Y+3] = d_in[x_max  + y_max2];
                    smem[BLOCK_X+3][BLOCK_Y+3] = d_in[x_max2 + y_max2];
                }
            }
            if (ty == 2)
            {
                smem[tx][0] = d_in[ix + y_min2];
                smem[tx][1] = d_in[ix + y_min];
                if (tx == BLOCK_X+1)
                {
                    smem[BLOCK_X+2][0] = d_in[x_max  + y_min2];
                    smem[BLOCK_X+3][0] = d_in[x_max2 + y_min2];
                    smem[BLOCK_X+2][1] = d_in[x_max  + y_min ];
                    smem[BLOCK_X+3][1] = d_in[x_max2 + y_min ];
                }
            }
            if (ty == BLOCK_Y+1)
            {
                smem[tx][BLOCK_Y+2] = d_in[ix + y_max];
                smem[tx][BLOCK_Y+3] = d_in[ix + y_max2];
                if (tx == 2)
                {
                    smem[0][BLOCK_X+2] = d_in[x_min2 + y_max ];
                    smem[0][BLOCK_X+3] = d_in[x_min2 + y_max2];
                    smem[1][BLOCK_X+2] = d_in[x_min  + y_max ];
                    smem[1][BLOCK_X+3] = d_in[x_min  + y_max2];
                }
            }
        }
        break;
    }
}

// Exchange trick: Morgan McGuire, ShaderX 2008
#define s2(a,b)            { float tmp = a; a = min(a,b); b = max(tmp,b); }
#define mn3(a,b,c)         s2(a,b); s2(a,c);
#define mx3(a,b,c)         s2(b,c); s2(a,c);

#define mnmx3(a,b,c)       mx3(a,b,c); s2(a,b);                               // 3 exchanges
#define mnmx4(a,b,c,d)     s2(a,b); s2(c,d); s2(a,c); s2(b,d);                // 4 exchanges
#define mnmx5(a,b,c,d,e)   s2(a,b); s2(c,d); mn3(a,c,e); mx3(b,d,e);          // 6 exchanges
#define mnmx6(a,b,c,d,e,f) s2(a,d); s2(b,e); s2(c,f); mn3(a,b,c); mx3(d,e,f); // 7 exchanges

__global__ void medfilt2_exch(int width, int height, float *d_out, float *d_in)
{
    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = ix + iy*width;

    if (ix >= width || iy >= height) return;

    // shared memory for 3x3 stencil
    __shared__ float smem[BLOCK_X+2][BLOCK_Y+2];

    // load in shared memory
    smem[tx][ty] = d_in[pos];
    set_bd_shared_memory<3>(smem, d_in, tx, ty, ix, iy, width, height);
    __syncthreads();

    // pull top six from shared memory
    float v[6] = { smem[tx-1][ty-1], smem[tx][ty-1], smem[tx+1][ty-1],
                   smem[tx-1][ty  ], smem[tx][ty  ], smem[tx+1][ty  ]};
 
    // with each pass, remove min and max values and add new value
    mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
    v[5] = smem[tx-1][ty+1]; // add new contestant
    mnmx5(v[1], v[2], v[3], v[4], v[5]);
    v[5] = smem[tx][ty+1];
    mnmx4(v[2], v[3], v[4], v[5]);
    v[5] = smem[tx+1][ty+1];
    mnmx3(v[3], v[4], v[5]);

    // pick the middle one
    d_out[pos] = v[4];
}


///////////////////////////////////////////////////////////////////////////////
/// \brief 3-by-3 median filtering of an image
/// \param[in]  img_in		input image
/// \param[in]  w			width of input image
/// \param[in]  h			height of input image
/// \param[out] img_out		output image
///////////////////////////////////////////////////////////////////////////////
static
void medfilt2(float *img_in, int w, int h, float *img_out)
{
    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

    medfilt2_exch<<<blocks,threads>>>(w, h, img_out, img_in);
    
}
