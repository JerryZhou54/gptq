#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<int M_TILE_SIZE, int N_TILE_SIZE, int K_TILE_SIZE>
__global__ void _nqmm_t(int *W, float *alpha, float *input, float *output, int M, int N, int K, int NUM_BITS);

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT =  24;
const int M_TILE_SIZE  = 128;
const int N_TILE_SIZE =  8;
const int K_TILE_SIZE =  8;

// ===========================================================

void lutGemm_cuda(
    torch::Tensor W,
    torch::Tensor alpha,
    torch::Tensor input,
    torch::Tensor output,
    int NUM_BITS
){

    assert(input.size(0) == 1);
    assert(output.size(0) == 1);

    int M = output.size(0);
    int N = output.size(1);
    int K = input.size(1);

    dim3 blocks(
        (K + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
        (N + BLOCKWIDTH - 1) / BLOCKWIDTH
    );
    dim3 threads(BLOCKWIDTH);

    _nqmm_t<M_TILE_SIZE, N_TILE_SIZE, K_TILE_SIZE><<<blocks, threads>>>(
        W.data_ptr<int>(),
        alpha.data_ptr<float>(),
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K, NUM_BITS
    );
}   


template<int M_TILE_SIZE, int N_TILE_SIZE, int K_TILE_SIZE>
__global__ void _nqmm_t(int *W, float *alpha, float *input, float *output, int M, int N, int K, int NUM_BITS){

    __shared__ float lut[K_TILE_SIZE/8][256][N_TILE_SIZE];

    // const int lut_y_size = K_TILE_SIZE/8;
    const int lut_x_size = blockDim.y / (K_TILE_SIZE/8);
 
    int lut_y = threadIdx.y/lut_x_size;
    int lut_x = threadIdx.y%lut_x_size;
    int lut_z = threadIdx.x;

    float *_inp = &input[lut_z*K + (blockIdx.y * K_TILE_SIZE + lut_y * 8) ];
    float base =    + (2 * ((lut_x>>0) & 1) - 1) * _inp[0]
                    + (2 * ((lut_x>>1) & 1) - 1) * _inp[1]
                    + (2 * ((lut_x>>2) & 1) - 1) * _inp[2]
                    + (2 * ((lut_x>>3) & 1) - 1) * _inp[3]
                    + (2 * ((lut_x>>4) & 1) - 1) * _inp[4]
                    + (2 * ((lut_x>>5) & 1) - 1) * _inp[5]
                    + (2 * ((lut_x>>6) & 1) - 1) * _inp[6]
                    + (2 * ((lut_x>>7) & 1) - 1) * _inp[7] ;
             
    lut[lut_y][lut_x][lut_z] = base;

    int s = (lut_x_size==1)  ?0:
            (lut_x_size==2)  ?1:
            (lut_x_size==4)  ?2:
            (lut_x_size==8)  ?3:
            (lut_x_size==16) ?4:
            (lut_x_size==32) ?5:
            (lut_x_size==64) ?6: 
            (lut_x_size==128)?7: 8;  
    for(;s<8;s++){
        float iValue =  2*_inp[s];
        for (int i = (1 << s); i < (1 << (s + 1)); i += lut_x_size) {
            lut[lut_y][i + lut_x][lut_z] =  lut[lut_y][i +  lut_x - (1 << s)][lut_z] + iValue;
        }
    }
    __syncthreads();

    int m_start = blockIdx.x * M_TILE_SIZE + threadIdx.y;
    int m_end = (blockIdx.x + 1) * M_TILE_SIZE;
    m_end = (m_end < M) ? m_end : M;
    int m_step = blockDim.y;

    int *bW = &W[blockIdx.y * K_TILE_SIZE/32 * NUM_BITS * M];
    float *_output = &output[lut_z * M];
    for(int m = m_start;m < m_end;m += m_step){
        float reg_o = 0;
        for(int b=0;b < NUM_BITS;b++){
            float   reg_a = alpha[b * M + m];
            float   reg_t_o = 0;
            for(int kt=0;kt < K_TILE_SIZE/32;kt++){
                uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m];
                int reg_w0 = (reg_w >> 8 * 0) & 255;   reg_t_o +=  + lut[kt*4 + 0][reg_w0][lut_z];
                int reg_w1 = (reg_w >> 8 * 1) & 255;   reg_t_o +=  + lut[kt*4 + 1][reg_w1][lut_z];
                int reg_w2 = (reg_w >> 8 * 2) & 255;   reg_t_o +=  + lut[kt*4 + 2][reg_w2][lut_z];
                int reg_w3 = (reg_w >> 8 * 3) & 255;   reg_t_o +=  + lut[kt*4 + 3][reg_w3][lut_z]; 
            }
            reg_o += reg_a * reg_t_o;
        }
        atomicAdd(&_output[m], reg_o);
    }
}
