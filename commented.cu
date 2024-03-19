/*
    解法来自 PKU HPCGAME 01 官方题解
    https://github.com/lcpu-club/hpcgame_1st_problems/blob/master/1st_g_conway3d/answer/answer.cu

    由我进行注释
*/




#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <iostream>

template <int size>
__global__ void life(uint32_t *datain, uint32_t *dataout, int xs, int ys, int zs)
{
    /*

    dim3 g = dim3(1, size, size / 4);
    dim3 b = dim3(size / 4, 1, 1);

        以 block 的 z 作为 y， block 的 y 作为 z。
        每个 thread 处理 y, y+1, y+2, y+3 中的一段深度

        BLOCK  +------------------>y
        (CELLs)|
               |    *
               |    *
               |    *
               |    *
               |
               ↓
               z

    */
    int y = blockIdx.z;

    __shared__ uint32_t tempsrc[(size / 4) * 4];
    __shared__ uint32_t tempin[(size / 4 + 2) * 6];

    /*
        BLOCK  +------------------>y
        (CELLs)|
               |    *
               |    *
               |    *
               |    *
               |
               ↓
               z

        LINE [y][z] =  -------- | -------- | -------- | -------- | ...
                        8bit        8bit       8bit       8bit     ...    size      *  8 bits
                                         32bit                      ---> (size / 4) * 32 bits

        tempsrc 保存从显存复制到 shared memory 的当前 （*) 的部分状态

    */

    uint8_t *tempsrc_ = (uint8_t *)tempsrc;
    // uint8_t* tempin_ = (uint8_t*)tempin;
    uint32_t tempout[4];
    /*
        每一个 uint32_t 保存 一个 z(BLOCK) 的末状态
    */

    int z = blockIdx.y;

    int z_[3];
    z_[0] = (z + zs - 1) % zs;
    z_[1] = z;
    z_[2] = (z + 1) % zs;

    /*
        BLOCK  +------------------>y
        (CELLs)|  0 1 2
               |
               |  ∷ ∷ ∷
               |  ∷ * ∷
               |  ∷ * ∷
               |  ∷ * ∷
               |  ∷ * ∷
               |  ∷ ∷ ∷
               ↓
               z

    */

    /*
        以下出现 thread 中的内部过程，即大小只考虑当前 thread，为 4cells * 4 depth

    */
    uint32_t loc[6]; //  thread var: 保存与目标列相邻的列状态 深度为 4cells = 32bit

    for (int j = 0; j < 6; j++) //
    {
        int y_ = y * 4 + ys - 1 + j;
        y_ = y_ % ys;
        /*
        j = 0: y_ = y * 4 + ys - 1
        j = 1: y_ = y * 4
        j = 2: y_ = y * 4 + 1
        j = 3: y_ = y * 4 + 2
        j = 4: y_ = y * 4 + 3
        j = 5: y_ = y * 4 + 4
        */

        loc[j] = 0;
        for (int k = 0; k < 3; k++)
        {
            int i_ = (z_[k] * ys + y_) * xs + threadIdx.x;

            loc[j] += datain[i_];
            /*
                        x = threadIdx.x (depth = 4*x (32bits))
                BLOCK  +------------------>y
                (CELLs)|k 0 1 2
                       |
                       |  ∷ ∷ ∷  --> sum -> loc[0]
                       |  ∷ ∷ ∷  --> sum -> loc[1]
                       |  ∷ ∷ ∷  --> sum -> loc[2]
                       |  ∷ ∷ ∷  --> sum -> loc[3]
                       |  ∷ ∷ ∷  --> sum -> loc[4]
                       |  ∷ ∷ ∷  --> sum -> loc[5]
                       ↓
                       z


                    loc[j] = datain[(    z_[0]     * ys + y_) * xs + threadIdx.x] +
                             datain[(    z_[1]     * ys + y_) * xs + threadIdx.x] +
                             datain[(    z_[2]     * ys + y_) * xs + threadIdx.x] ;


                    loc[j] = -------- | -------- | -------- | -------- +
                             cell(8b)   cell(8b)   cell(8b)   cell(8b)
                             -------- | -------- | -------- | -------- +

                             -------- | -------- | -------- | -------- ;

                    加起来就得到对应的活细胞总数，同时也保留了位置（深度 x）的信息

            */
        }
    }

    for (int j = 0; j < 4; j++)
    {
        int y_ = y * 4 + j;

        int i_ = (z * ys + y_) * xs + threadIdx.x;
        uint32_t loc = datain[i_];

        int ini = j * (size / 4) + threadIdx.x;
        tempsrc[ini] = loc;
    }

    /*
            x = threadIdx.x (depth = 4*x (32bits))
        BLOCK  +------------------>y
        (CELLs)|
               |
               | j
               | 0  *    --> tmpsrc[0 * (size / 4) + threadIdx.x]
               | 1  *    --> tmpsrc[1 * (size / 4) + threadIdx.x]
               | 2  *    --> tmpsrc[2 * (size / 4) + threadIdx.x]
               | 3  *    --> tmpsrc[3 * (size / 4) + threadIdx.x]
               |
               ↓
               z

        读入当前细胞状态
         tmpsrc =  -> x (depth)
           | 0    -------------------------------- | -------------------------------- | ...
           |             32bits 4cells
           | 1    -------------------------------- | -------------------------------- | ...
           |
           | 2    -------------------------------- | -------------------------------- | ...
           |
           | 3    -------------------------------- | -------------------------------- | ...
           ↓
           z = 0, 1, 2, 3
    */
    __syncthreads(); // 同步 shared memory， 等待所有线程完成读取
    // 此时所有的 tmpsrc 都应该被填充

    loc[0] = loc[0] + loc[1] + loc[2];
    loc[1] = loc[1] + loc[2] + loc[3];
    loc[2] = loc[2] + loc[3] + loc[4];
    loc[3] = loc[3] + loc[4] + loc[5];
    /*
            x = threadIdx.x (depth = 4*x (32bits))
    BLOCK  +------------------>y
    (CELLs)|
           |
           |  ∷ ∷ ∷  --> sum -> loc[0] ⎤
           |  ∷ * ∷  --> sum -> loc[1] ⎥ -> loc[0] * 附近的存活细胞数量（包括自身）
           |  ∷ * ∷  --> sum -> loc[2] ⎦
           |  ∷ * ∷  --> sum -> loc[3]
           |  ∷ * ∷  --> sum -> loc[4]
           |  ∷ ∷ ∷  --> sum -> loc[5]
           ↓
           z
    */

    tempin[0 * (size / 4 + 2) + threadIdx.x + 1] = loc[0];
    tempin[1 * (size / 4 + 2) + threadIdx.x + 1] = loc[1];
    tempin[2 * (size / 4 + 2) + threadIdx.x + 1] = loc[2];
    tempin[3 * (size / 4 + 2) + threadIdx.x + 1] = loc[3];
    /*
         tmpin =  -> x (depth)
           | 0    -------------------------------- # -------------------------------- | -------------------------------- | ... # --------------------------------
           |             为循环空间保留                                                                                                      为循环空间保留
           | 1    -------------------------------- # -------------------------------- | -------------------------------- | ... # --------------------------------
           |                                             32bits ALIVE CELL COUNT
           | 2    -------------------------------- # -------------------------------- | -------------------------------- | ... # --------------------------------
           |
           | 3    -------------------------------- # -------------------------------- | -------------------------------- | ... # --------------------------------
           ↓
           z = 0, 1, 2, 3
    */
    __syncthreads();
    // 此时所有的 tmpin 都应该被填充 （除了首尾）

    if (threadIdx.x < 4)
    {
        tempin[threadIdx.x * (size / 4 + 2) + 0] = tempin[threadIdx.x * (size / 4 + 2) + (size / 4)];
        tempin[threadIdx.x * (size / 4 + 2) + (size / 4 + 1)] = tempin[threadIdx.x * (size / 4 + 2) + 1];
    } // 填充首尾部分

    /*
         tmpin =  -> x (depth)
           | 0    -------------------------------- # -------------------------------- | -------------------------------- | ... --------------------------------  # --------------------------------
           |             =(2)                                  (1)                                                                         (2)                                =(1)
           | 1    -------------------------------- # -------------------------------- | -------------------------------- | ... --------------------------------  # --------------------------------
           |                                             32bits ALIVE CELL COUNT
           | 2    -------------------------------- # -------------------------------- | -------------------------------- | ... --------------------------------  # --------------------------------
           |
           | 3    -------------------------------- # -------------------------------- | -------------------------------- | ... --------------------------------  # --------------------------------
           ↓
           z = 0, 1, 2, 3
    */
    __syncthreads();
    // 此时所有的 tmpin 都应该被填充

    for (int j = 0; j < 4; j++) // 逐列处理该线程负责的 4 列中的部分细胞
    {
        uint32_t loc0 = tempin[j * (size / 4 + 2) + threadIdx.x];
        uint32_t loc1 = tempin[j * (size / 4 + 2) + threadIdx.x + 1];
        uint32_t loc2 = tempin[j * (size / 4 + 2) + threadIdx.x + 2];

        /*
            对于每一列 取出相邻的深度
             loc =  -> x (depth)
               | 0  --------------------------------  1  --------------------------------  2  --------------------------------
               |             prev_block                     MY_block(for this thread)                     next_block
               |
               ↓
        */
        loc0 = loc1 + (loc1 >> 8) + (loc2 << 24) + (loc1 << 8) + (loc0 >> 24);

        /*
            对于 prev_block 和 next_block 只需要取其一部分（底部和顶部）
            这里的数据都是 小端
             loc =  -> x (depth)
               | 0  -------- | -------- | -------- | --------  1   -------- | -------- | -------- | --------  2   -------- | -------- | -------- | --------
               |        1          2         3           4            5           6         7           8             9         A           B         C
               |
               |             prev_block                                   MY_block(for this thread)                       next_block
               |
               ↓

            loc0 =  -------- | -------- | -------- | --------  loc1 +
                        5          6         7           8
                    -------- | -------- | -------- | 00000000  loc1 >> 8  +
                        6          7         8
                    00000000 | 00000000 | 00000000 | --------  loc2 << 24 +
                                                         9
                    00000000 | -------- | -------- | --------  loc1 << 8  +
                                   5         6           7
                    -------- | 00000000 | 00000000 | 00000000  loc0 >> 24
                        4
                        ⇩          ⇩          ⇩         ⇩
                    depth=0     depth=1    depth=2   depth=3

                       (1)         (2)       (3)        (4)


        */
        for (int step = 0; step < 4; step++) // 处理一列中的深度
        {

            uint32_t c = loc0 & 0xff;
            /*

                c 的取值顺序是  (1) (2) (3) (4)

                分别对应于对应深度的活细胞数量

            */

            loc0 = loc0 >> 8;

            uint32_t loc = tempsrc_[(j * (size / 4) + threadIdx.x) * 4 + step]; // 获取当前细胞状态
            if (c == 6 || 6 <= c && c <= 8 && loc)                              // 应用生命游戏规则
            {
                tempout[step] = 1;
            }
            else
            {
                tempout[step] = 0;
            }
        }

        uint32_t out = tempout[0] + (tempout[1] << 8) + (tempout[2] << 16) + (tempout[3] << 24);
        /*

                out  =  -------- | 00000000 | 00000000 | 00000000  tempout[0]       +
                          bool
                        00000000 | -------- | 00000000 | 00000000  tempout[1] << 8  +

                        00000000 | 00000000 | -------- | 00000000  tempout[2] << 16 +

                        00000000 | 00000000 | 00000000 | --------  tempout[3] << 24 +

        */

        int y_ = y * 4 + j;
        int i_ = (z * ys + y_) * xs + threadIdx.x;
        dataout[i_] = out;

        /*
            x(depth) = threadIdx.x

            dataout  =  -------- | -------- | -------- | -------- | ...
                           8bit        8bit       8bit       8bit     ...    size      *  8 bits
                                             32bit                      ---> (size / 4) * 32 bits

        */
    }
}

void tofile(void *p, size_t n, const char *fn)
{
    FILE *fi = fopen(fn, "wb");
    fwrite(p, 1, n, fi);
    fclose(fi);
}

void fromfile(void *p, size_t n, const char *fn)
{
    FILE *fi = fopen(fn, "rb");
    fread(p, 1, n, fi);
    fclose(fi);
}

int main(int argc, const char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " <input_path> <output_path> <N>" << std::endl;

        argv = (const char **)malloc(32);
        argv[1] = "../../conf.data";
        argv[2] = "../../out.data";
        argv[3] = "2";
    }

    int itn;
    sscanf(argv[3], "%d", &itn);

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addWithCuda failed!");
    }

    uint32_t *a;
    uint32_t *dev_a = 0;
    uint32_t *dev_b = 0;

    int64_t size;
    int64_t t;

    FILE *fi = fopen(argv[1], "rb");
    fread(&size, 1, 8, fi);
    fread(&t, 1, 8, fi);
    a = (uint32_t *)malloc(size * size * size);
    fread(a, 1, size * size * size, fi);
    fclose(fi);

    cudaStatus = cudaMalloc((void **)&dev_a, size * size * size);
    /*

    每个细胞占用 8bit，即一个uint8_t

    */
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void **)&dev_b, size * size * size);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMemcpy(dev_a, a, size * size * size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "copy failed!");
    }

    dim3 g = dim3(1, size, size / 4);
    /*

        将空间分为 size * (size / 4) 个 block。
        每个 block 处理此 y 坐标 和 起始 z 坐标下下 1 * 4 列细胞

    */

    dim3 b = dim3(size / 4, 1, 1);

    /*

        每个 thread 处理  4 列 * 4深 个细胞。

    */

    auto t1 = std::chrono::steady_clock::now();

    /*

        进入核函数

    */

    if (size == 256)
    {
        for (int i = 0; i < itn / 2; i++)
        {
            /*
                避免交换指针
            */
            life<256><<<g, b>>>(dev_a, dev_b, size / 4, size, size);
            life<256><<<g, b>>>(dev_b, dev_a, size / 4, size, size);
        }
    }
    else if (size == 512)
    {
        for (int i = 0; i < itn / 2; i++)
        {
            life<512><<<g, b>>>(dev_a, dev_b, size / 4, size, size);
            life<512><<<g, b>>>(dev_b, dev_a, size / 4, size, size);
        }
    }
    else if (size == 1024)
    {
        for (int i = 0; i < itn / 2; i++)
        {
            life<1024><<<g, b>>>(dev_a, dev_b, size / 4, size, size);
            life<1024><<<g, b>>>(dev_b, dev_a, size / 4, size, size);
        }
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "calc failed");
    }
    auto t2 = std::chrono::steady_clock::now();

    cudaStatus = cudaMemcpy(a, dev_a, size * size * size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    t += itn;
    fi = fopen(argv[2], "wb");
    fwrite(&size, 1, 8, fi);
    fwrite(&t, 1, 8, fi);
    fwrite(a, 1, size * size * size, fi);
    fclose(fi);

    int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("%d\n", d1);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    return 0;
}
