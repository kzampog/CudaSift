//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//  

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>

#include <cuda_sift/cudautils.h>
#include <cuda_sift/cudaImage.h>
#include <cuda_sift/cudaSift.h>
#include <cuda_sift/cudaSiftD.h>
#include <cuda_sift/cudaSiftH.h>
#include <driver_types.h>


///////////////////////////////////////////////////////////////////////////////
// Kernel configuration
///////////////////////////////////////////////////////////////////////////////

__constant__ int d_MaxNumPoints;
__device__ unsigned int d_PointCounter[8*2+1];
__constant__ float d_ScaleDownKernel[5];
__constant__ float d_LowPassKernel[2*LOWPASS_R+1];
__constant__ float d_LaplaceKernel[8*12*16];

///////////////////////////////////////////////////////////////////////////////
// Lowpass filter and subsample image
///////////////////////////////////////////////////////////////////////////////
__global__ void ScaleDown(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
    __shared__ float inrow[SCALEDOWN_W+4];
    __shared__ float brow[5*(SCALEDOWN_W/2)];
    __shared__ int yRead[SCALEDOWN_H+4];
    __shared__ int yWrite[SCALEDOWN_H+4];
#define dx2 (SCALEDOWN_W/2)
    const int tx = threadIdx.x;
    const int tx0 = tx + 0*dx2;
    const int tx1 = tx + 1*dx2;
    const int tx2 = tx + 2*dx2;
    const int tx3 = tx + 3*dx2;
    const int tx4 = tx + 4*dx2;
    const int xStart = blockIdx.x*SCALEDOWN_W;
    const int yStart = blockIdx.y*SCALEDOWN_H;
    const int xWrite = xStart/2 + tx;
    const float *k = d_ScaleDownKernel;
    if (tx<SCALEDOWN_H+4) {
        int y = yStart + tx - 1;
        y = (y<0 ? 0 : y);
        y = (y>=height ? height-1 : y);
        yRead[tx] = y*pitch;
        yWrite[tx] = (yStart + tx - 4)/2 * newpitch;
    }
    __syncthreads();
    int xRead = xStart + tx - 2;
    xRead = (xRead<0 ? 0 : xRead);
    xRead = (xRead>=width ? width-1 : xRead);
    for (int dy=0;dy<SCALEDOWN_H+4;dy+=5) {
        inrow[tx] = d_Data[yRead[dy+0] + xRead];
        __syncthreads();
        if (tx<dx2)
            brow[tx0] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
        __syncthreads();
        if (tx<dx2 && dy>=4 && !(dy&1))
            d_Result[yWrite[dy+0] + xWrite] = k[2]*brow[tx2] + k[0]*(brow[tx0]+brow[tx4]) + k[1]*(brow[tx1]+brow[tx3]);
        if (dy<(SCALEDOWN_H+3)) {
            inrow[tx] = d_Data[yRead[dy+1] + xRead];
            __syncthreads();
            if (tx<dx2)
                brow[tx1] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
            __syncthreads();
            if (tx<dx2 && dy>=3 && (dy&1))
                d_Result[yWrite[dy+1] + xWrite] = k[2]*brow[tx3] + k[0]*(brow[tx1]+brow[tx0]) + k[1]*(brow[tx2]+brow[tx4]);
        }
        if (dy<(SCALEDOWN_H+2)) {
            inrow[tx] = d_Data[yRead[dy+2] + xRead];
            __syncthreads();
            if (tx<dx2)
                brow[tx2] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
            __syncthreads();
            if (tx<dx2 && dy>=2 && !(dy&1))
                d_Result[yWrite[dy+2] + xWrite] = k[2]*brow[tx4] + k[0]*(brow[tx2]+brow[tx1]) + k[1]*(brow[tx3]+brow[tx0]);
        }
        if (dy<(SCALEDOWN_H+1)) {
            inrow[tx] = d_Data[yRead[dy+3] + xRead];
            __syncthreads();
            if (tx<dx2)
                brow[tx3] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
            __syncthreads();
            if (tx<dx2 && dy>=1 && (dy&1))
                d_Result[yWrite[dy+3] + xWrite] = k[2]*brow[tx0] + k[0]*(brow[tx3]+brow[tx2]) + k[1]*(brow[tx4]+brow[tx1]);
        }
        if (dy<SCALEDOWN_H) {
            inrow[tx] = d_Data[yRead[dy+4] + xRead];
            __syncthreads();
            if (tx<dx2)
                brow[tx4] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
            __syncthreads();
            if (tx<dx2 && !(dy&1))
                d_Result[yWrite[dy+4] + xWrite] = k[2]*brow[tx1] + k[0]*(brow[tx4]+brow[tx3]) + k[1]*(brow[tx0]+brow[tx2]);
        }
        __syncthreads();
    }
}

__global__ void ScaleUp(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
#define BW (SCALEUP_W/2 + 2)
#define BH (SCALEUP_H/2 + 2)
    __shared__ float buffer[BW*BH];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    if (tx<BW && ty<BH) {
        int x = min(max(blockIdx.x*(SCALEUP_W/2) + tx - 1, 0), width-1);
        int y = min(max(blockIdx.y*(SCALEUP_H/2) + ty - 1, 0), height-1);
        buffer[ty*BW + tx] = d_Data[y*pitch + x];
    }
    __syncthreads();
    int x = blockIdx.x*SCALEUP_W + tx;
    int y = blockIdx.y*SCALEUP_H + ty;
    if (x<2*width && y<2*height) {
        int bx = (tx + 1)/2;
        int by = (ty + 1)/2;
        int bp = by*BW + bx;
        float wx = 0.25f + (tx&1)*0.50f;
        float wy = 0.25f + (ty&1)*0.50f;
        d_Result[y*newpitch + x] = wy*(wx*buffer[bp] + (1.0f-wx)*buffer[bp+1]) +
                                   (1.0f-wy)*(wx*buffer[bp+BW] + (1.0f-wx)*buffer[bp+BW+1]);
    }
}

__global__ void ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftPoint *d_sift, int fstPts, float subsampling)
{
    __shared__ float gauss[16];
    __shared__ float buffer[128];
    __shared__ float sums[4];

    const int tx = threadIdx.x; // 0 -> 16
    const int ty = threadIdx.y; // 0 -> 8
    const int idx = ty*16 + tx;
    const int bx = blockIdx.x + fstPts;  // 0 -> numPts
    if (ty==0)
        gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);
    buffer[idx] = 0.0f;
    __syncthreads();

    // Compute angles and gradients
    float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
    float sina = sinf(theta);           // cosa -sina
    float cosa = cosf(theta);           // sina  cosa
    float scale = 12.0f/16.0f*d_sift[bx].scale;
    float ssina = scale*sina;
    float scosa = scale*cosa;

    for (int y=ty;y<16;y+=8) {
        float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina;
        float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa;
        float dx = tex2D<float>(texObj, xpos+cosa, ypos+sina) -
                   tex2D<float>(texObj, xpos-cosa, ypos-sina);
        float dy = tex2D<float>(texObj, xpos-sina, ypos+cosa) -
                   tex2D<float>(texObj, xpos+sina, ypos-cosa);
        float grad = gauss[y]*gauss[tx] * sqrtf(dx*dx + dy*dy);
        float angf = 4.0f/3.1415f*atan2f(dy, dx) + 4.0f;

        int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins
        float horf = (tx - 1.5f)/4.0f - hori;
        float ihorf = 1.0f - horf;
        int veri = (y + 2)/4 - 1;
        float verf = (y - 1.5f)/4.0f - veri;
        float iverf = 1.0f - verf;
        int angi = angf;
        int angp = (angi<7 ? angi+1 : 0);
        angf -= angi;
        float iangf = 1.0f - angf;

        int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated
        int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
        int p2 = angp + hist;
        if (tx>=2) {
            float grad1 = ihorf*grad;
            if (y>=2) {   // Upper left
                float grad2 = iverf*grad1;
                atomicAdd(buffer + p1, iangf*grad2);
                atomicAdd(buffer + p2,  angf*grad2);
            }
            if (y<=13) {  // Lower left
                float grad2 = verf*grad1;
                atomicAdd(buffer + p1+32, iangf*grad2);
                atomicAdd(buffer + p2+32,  angf*grad2);
            }
        }
        if (tx<=13) {
            float grad1 = horf*grad;
            if (y>=2) {    // Upper right
                float grad2 = iverf*grad1;
                atomicAdd(buffer + p1+8, iangf*grad2);
                atomicAdd(buffer + p2+8,  angf*grad2);
            }
            if (y<=13) {   // Lower right
                float grad2 = verf*grad1;
                atomicAdd(buffer + p1+40, iangf*grad2);
                atomicAdd(buffer + p2+40,  angf*grad2);
            }
        }
    }
    __syncthreads();

    // Normalize twice and suppress peaks first time
    float sum = buffer[idx]*buffer[idx];
    for (int i=16;i>0;i/=2)
        sum += ShiftDown(sum, i);
    if ((idx&31)==0)
        sums[idx/32] = sum;
    __syncthreads();
    float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
    tsum1 = min(buffer[idx] * rsqrtf(tsum1), 0.2f);

    sum = tsum1*tsum1;
    for (int i=16;i>0;i/=2)
        sum += ShiftDown(sum, i);
    if ((idx&31)==0)
        sums[idx/32] = sum;
    __syncthreads();

    float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
    float *desc = d_sift[bx].data;
    desc[idx] = tsum1 * rsqrtf(tsum2);
    if (idx==0) {
        d_sift[bx].xpos *= subsampling;
        d_sift[bx].ypos *= subsampling;
        d_sift[bx].scale *= subsampling;
    }
}


__global__ void ExtractSiftDescriptorsCONST(cudaTextureObject_t texObj, SiftPoint *d_sift, float subsampling, int octave)
{
    __shared__ float gauss[16];
    __shared__ float buffer[128];
    __shared__ float sums[4];

    const int tx = threadIdx.x; // 0 -> 16
    const int ty = threadIdx.y; // 0 -> 8
    const int idx = ty*16 + tx;
    if (ty==0)
        gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);

    int fstPts = min(d_PointCounter[2*octave-1], d_MaxNumPoints);
    int totPts = min(d_PointCounter[2*octave+1], d_MaxNumPoints);
    //if (tx==0 && ty==0)
    //  printf("%d %d %d %d\n", octave, fstPts, min(d_PointCounter[2*octave], d_MaxNumPoints), totPts);
    for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {

        buffer[idx] = 0.0f;
        __syncthreads();

        // Compute angles and gradients
        float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
        float sina = sinf(theta);           // cosa -sina
        float cosa = cosf(theta);           // sina  cosa
        float scale = 12.0f/16.0f*d_sift[bx].scale;
        float ssina = scale*sina;
        float scosa = scale*cosa;

        for (int y=ty;y<16;y+=8) {
            float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina;
            float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa;
            float dx = tex2D<float>(texObj, xpos+cosa, ypos+sina) -
                       tex2D<float>(texObj, xpos-cosa, ypos-sina);
            float dy = tex2D<float>(texObj, xpos-sina, ypos+cosa) -
                       tex2D<float>(texObj, xpos+sina, ypos-cosa);
            float grad = gauss[y]*gauss[tx] * sqrtf(dx*dx + dy*dy);
            float angf = 4.0f/3.1415f*atan2f(dy, dx) + 4.0f;

            int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins
            float horf = (tx - 1.5f)/4.0f - hori;
            float ihorf = 1.0f - horf;
            int veri = (y + 2)/4 - 1;
            float verf = (y - 1.5f)/4.0f - veri;
            float iverf = 1.0f - verf;
            int angi = angf;
            int angp = (angi<7 ? angi+1 : 0);
            angf -= angi;
            float iangf = 1.0f - angf;

            int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated
            int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
            int p2 = angp + hist;
            if (tx>=2) {
                float grad1 = ihorf*grad;
                if (y>=2) {   // Upper left
                    float grad2 = iverf*grad1;
                    atomicAdd(buffer + p1, iangf*grad2);
                    atomicAdd(buffer + p2,  angf*grad2);
                }
                if (y<=13) {  // Lower left
                    float grad2 = verf*grad1;
                    atomicAdd(buffer + p1+32, iangf*grad2);
                    atomicAdd(buffer + p2+32,  angf*grad2);
                }
            }
            if (tx<=13) {
                float grad1 = horf*grad;
                if (y>=2) {    // Upper right
                    float grad2 = iverf*grad1;
                    atomicAdd(buffer + p1+8, iangf*grad2);
                    atomicAdd(buffer + p2+8,  angf*grad2);
                }
                if (y<=13) {   // Lower right
                    float grad2 = verf*grad1;
                    atomicAdd(buffer + p1+40, iangf*grad2);
                    atomicAdd(buffer + p2+40,  angf*grad2);
                }
            }
        }
        __syncthreads();

        // Normalize twice and suppress peaks first time
        float sum = buffer[idx]*buffer[idx];
        for (int i=16;i>0;i/=2)
            sum += ShiftDown(sum, i);
        if ((idx&31)==0)
            sums[idx/32] = sum;
        __syncthreads();
        float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
        tsum1 = min(buffer[idx] * rsqrtf(tsum1), 0.2f);

        sum = tsum1*tsum1;
        for (int i=16;i>0;i/=2)
            sum += ShiftDown(sum, i);
        if ((idx&31)==0)
            sums[idx/32] = sum;
        __syncthreads();

        float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
        float *desc = d_sift[bx].data;
        desc[idx] = tsum1 * rsqrtf(tsum2);
        if (idx==0) {
            d_sift[bx].xpos *= subsampling;
            d_sift[bx].ypos *= subsampling;
            d_sift[bx].scale *= subsampling;
        }
        __syncthreads();
    }
}


__global__ void ExtractSiftDescriptorsOld(cudaTextureObject_t texObj, SiftPoint *d_sift, int fstPts, float subsampling)
{
    __shared__ float gauss[16];
    __shared__ float buffer[128];
    __shared__ float sums[128];

    const int tx = threadIdx.x; // 0 -> 16
    const int ty = threadIdx.y; // 0 -> 8
    const int idx = ty*16 + tx;
    const int bx = blockIdx.x + fstPts;  // 0 -> numPts
    if (ty==0)
        gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);
    buffer[idx] = 0.0f;
    __syncthreads();

    // Compute angles and gradients
    float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
    float sina = sinf(theta);           // cosa -sina
    float cosa = cosf(theta);           // sina  cosa
    float scale = 12.0f/16.0f*d_sift[bx].scale;
    float ssina = scale*sina;
    float scosa = scale*cosa;

    for (int y=ty;y<16;y+=8) {
        float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina;
        float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa;
        float dx = tex2D<float>(texObj, xpos+cosa, ypos+sina) -
                   tex2D<float>(texObj, xpos-cosa, ypos-sina);
        float dy = tex2D<float>(texObj, xpos-sina, ypos+cosa) -
                   tex2D<float>(texObj, xpos+sina, ypos-cosa);
        float grad = gauss[y]*gauss[tx] * sqrtf(dx*dx + dy*dy);
        float angf = 4.0f/3.1415f*atan2f(dy, dx) + 4.0f;

        int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins
        float horf = (tx - 1.5f)/4.0f - hori;
        float ihorf = 1.0f - horf;
        int veri = (y + 2)/4 - 1;
        float verf = (y - 1.5f)/4.0f - veri;
        float iverf = 1.0f - verf;
        int angi = angf;
        int angp = (angi<7 ? angi+1 : 0);
        angf -= angi;
        float iangf = 1.0f - angf;

        int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated
        int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
        int p2 = angp + hist;
        if (tx>=2) {
            float grad1 = ihorf*grad;
            if (y>=2) {   // Upper left
                float grad2 = iverf*grad1;
                atomicAdd(buffer + p1, iangf*grad2);
                atomicAdd(buffer + p2,  angf*grad2);
            }
            if (y<=13) {  // Lower left
                float grad2 = verf*grad1;
                atomicAdd(buffer + p1+32, iangf*grad2);
                atomicAdd(buffer + p2+32,  angf*grad2);
            }
        }
        if (tx<=13) {
            float grad1 = horf*grad;
            if (y>=2) {    // Upper right
                float grad2 = iverf*grad1;
                atomicAdd(buffer + p1+8, iangf*grad2);
                atomicAdd(buffer + p2+8,  angf*grad2);
            }
            if (y<=13) {   // Lower right
                float grad2 = verf*grad1;
                atomicAdd(buffer + p1+40, iangf*grad2);
                atomicAdd(buffer + p2+40,  angf*grad2);
            }
        }
    }
    __syncthreads();

    // Normalize twice and suppress peaks first time
    if (idx<64)
        sums[idx] = buffer[idx]*buffer[idx] + buffer[idx+64]*buffer[idx+64];
    __syncthreads();
    if (idx<32) sums[idx] = sums[idx] + sums[idx+32];
    __syncthreads();
    if (idx<16) sums[idx] = sums[idx] + sums[idx+16];
    __syncthreads();
    if (idx<8)  sums[idx] = sums[idx] + sums[idx+8];
    __syncthreads();
    if (idx<4)  sums[idx] = sums[idx] + sums[idx+4];
    __syncthreads();
    float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
    buffer[idx] = buffer[idx] * rsqrtf(tsum1);

    if (buffer[idx]>0.2f)
        buffer[idx] = 0.2f;
    __syncthreads();
    if (idx<64)
        sums[idx] = buffer[idx]*buffer[idx] + buffer[idx+64]*buffer[idx+64];
    __syncthreads();
    if (idx<32) sums[idx] = sums[idx] + sums[idx+32];
    __syncthreads();
    if (idx<16) sums[idx] = sums[idx] + sums[idx+16];
    __syncthreads();
    if (idx<8)  sums[idx] = sums[idx] + sums[idx+8];
    __syncthreads();
    if (idx<4)  sums[idx] = sums[idx] + sums[idx+4];
    __syncthreads();
    float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];

    float *desc = d_sift[bx].data;
    desc[idx] = buffer[idx] * rsqrtf(tsum2);
    if (idx==0) {
        d_sift[bx].xpos *= subsampling;
        d_sift[bx].ypos *= subsampling;
        d_sift[bx].scale *= subsampling;
    }
}


__global__ void RescalePositions(SiftPoint *d_sift, int numPts, float scale)
{
    int num = blockIdx.x*blockDim.x + threadIdx.x;
    if (num<numPts) {
        d_sift[num].xpos *= scale;
        d_sift[num].ypos *= scale;
        d_sift[num].scale *= scale;
    }
}


__global__ void ComputeOrientations(cudaTextureObject_t texObj, SiftPoint *d_Sift, int fstPts)
{
    __shared__ float hist[64];
    __shared__ float gauss[11];
    const int tx = threadIdx.x;
    const int bx = blockIdx.x + fstPts;
    float i2sigma2 = -1.0f/(4.5f*d_Sift[bx].scale*d_Sift[bx].scale);
    if (tx<11)
        gauss[tx] = exp(i2sigma2*(tx-5)*(tx-5));
    if (tx<64)
        hist[tx] = 0.0f;
    __syncthreads();
    float xp = d_Sift[bx].xpos - 5.0f;
    float yp = d_Sift[bx].ypos - 5.0f;
    int yd = tx/11;
    int xd = tx - yd*11;
    float xf = xp + xd;
    float yf = yp + yd;
    if (yd<11) {
        float dx = tex2D<float>(texObj, xf+1.0, yf) - tex2D<float>(texObj, xf-1.0, yf);
        float dy = tex2D<float>(texObj, xf, yf+1.0) - tex2D<float>(texObj, xf, yf-1.0);
        int bin = 16.0f*atan2f(dy, dx)/3.1416f + 16.5f;
        if (bin>31)
            bin = 0;
        float grad = sqrtf(dx*dx + dy*dy);
        atomicAdd(&hist[bin], grad*gauss[xd]*gauss[yd]);
    }
    __syncthreads();
    int x1m = (tx>=1 ? tx-1 : tx+31);
    int x1p = (tx<=30 ? tx+1 : tx-31);
    if (tx<32) {
        int x2m = (tx>=2 ? tx-2 : tx+30);
        int x2p = (tx<=29 ? tx+2 : tx-30);
        hist[tx+32] = 6.0f*hist[tx] + 4.0f*(hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
    }
    __syncthreads();
    if (tx<32) {
        float v = hist[32+tx];
        hist[tx] = (v>hist[32+x1m] && v>=hist[32+x1p] ? v : 0.0f);
    }
    __syncthreads();
    if (tx==0) {
        float maxval1 = 0.0;
        float maxval2 = 0.0;
        int i1 = -1;
        int i2 = -1;
        for (int i=0;i<32;i++) {
            float v = hist[i];
            if (v>maxval1) {
                maxval2 = maxval1;
                maxval1 = v;
                i2 = i1;
                i1 = i;
            } else if (v>maxval2) {
                maxval2 = v;
                i2 = i;
            }
        }
        float val1 = hist[32+((i1+1)&31)];
        float val2 = hist[32+((i1+31)&31)];
        float peak = i1 + 0.5f*(val1-val2) / (2.0f*maxval1-val1-val2);
        d_Sift[bx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
        if (maxval2>0.8f*maxval1) {
            float val1 = hist[32+((i2+1)&31)];
            float val2 = hist[32+((i2+31)&31)];
            float peak = i2 + 0.5f*(val1-val2) / (2.0f*maxval2-val1-val2);
            unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
            if (idx<d_MaxNumPoints) {
                d_Sift[idx].xpos = d_Sift[bx].xpos;
                d_Sift[idx].ypos = d_Sift[bx].ypos;
                d_Sift[idx].scale = d_Sift[bx].scale;
                d_Sift[idx].sharpness = d_Sift[bx].sharpness;
                d_Sift[idx].edgeness = d_Sift[bx].edgeness;
                d_Sift[idx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);;
                d_Sift[idx].subsampling = d_Sift[bx].subsampling;
            }
        }
    }
}

// With constant number of blocks
__global__ void ComputeOrientationsCONST(cudaTextureObject_t texObj, SiftPoint *d_Sift, int octave)
{
    __shared__ float hist[64];
    __shared__ float gauss[11];
    const int tx = threadIdx.x;

    int fstPts = min(d_PointCounter[2*octave-1], d_MaxNumPoints);
    int totPts = min(d_PointCounter[2*octave+0], d_MaxNumPoints);
    for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {

        float i2sigma2 = -1.0f/(4.5f*d_Sift[bx].scale*d_Sift[bx].scale);
        if (tx<11)
            gauss[tx] = exp(i2sigma2*(tx-5)*(tx-5));
        if (tx<64)
            hist[tx] = 0.0f;
        __syncthreads();
        float xp = d_Sift[bx].xpos - 5.0f;
        float yp = d_Sift[bx].ypos - 5.0f;
        int yd = tx/11;
        int xd = tx - yd*11;
        float xf = xp + xd;
        float yf = yp + yd;
        if (yd<11) {
            float dx = tex2D<float>(texObj, xf+1.0, yf) - tex2D<float>(texObj, xf-1.0, yf);
            float dy = tex2D<float>(texObj, xf, yf+1.0) - tex2D<float>(texObj, xf, yf-1.0);
            int bin = 16.0f*atan2f(dy, dx)/3.1416f + 16.5f;
            if (bin>31)
                bin = 0;
            float grad = sqrtf(dx*dx + dy*dy);
            atomicAdd(&hist[bin], grad*gauss[xd]*gauss[yd]);
        }
        __syncthreads();
        int x1m = (tx>=1 ? tx-1 : tx+31);
        int x1p = (tx<=30 ? tx+1 : tx-31);
        if (tx<32) {
            int x2m = (tx>=2 ? tx-2 : tx+30);
            int x2p = (tx<=29 ? tx+2 : tx-30);
            hist[tx+32] = 6.0f*hist[tx] + 4.0f*(hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
        }
        __syncthreads();
        if (tx<32) {
            float v = hist[32+tx];
            hist[tx] = (v>hist[32+x1m] && v>=hist[32+x1p] ? v : 0.0f);
        }
        __syncthreads();
        if (tx==0) {
            float maxval1 = 0.0;
            float maxval2 = 0.0;
            int i1 = -1;
            int i2 = -1;
            for (int i=0;i<32;i++) {
                float v = hist[i];
                if (v>maxval1) {
                    maxval2 = maxval1;
                    maxval1 = v;
                    i2 = i1;
                    i1 = i;
                } else if (v>maxval2) {
                    maxval2 = v;
                    i2 = i;
                }
            }
            float val1 = hist[32+((i1+1)&31)];
            float val2 = hist[32+((i1+31)&31)];
            float peak = i1 + 0.5f*(val1-val2) / (2.0f*maxval1-val1-val2);
            d_Sift[bx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
            atomicMax(&d_PointCounter[2*octave+1], d_PointCounter[2*octave+0]);
            if (maxval2>0.8f*maxval1) {
                float val1 = hist[32+((i2+1)&31)];
                float val2 = hist[32+((i2+31)&31)];
                float peak = i2 + 0.5f*(val1-val2) / (2.0f*maxval2-val1-val2);
                unsigned int idx = atomicInc(&d_PointCounter[2*octave+1], 0x7fffffff);
                if (idx<d_MaxNumPoints) {
                    d_Sift[idx].xpos = d_Sift[bx].xpos;
                    d_Sift[idx].ypos = d_Sift[bx].ypos;
                    d_Sift[idx].scale = d_Sift[bx].scale;
                    d_Sift[idx].sharpness = d_Sift[bx].sharpness;
                    d_Sift[idx].edgeness = d_Sift[bx].edgeness;
                    d_Sift[idx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);;
                    d_Sift[idx].subsampling = d_Sift[bx].subsampling;
                }
            }
        }
        __syncthreads();
    }
}


#if 0
__global__ void OrientAndExtract(cudaTextureObject_t texObj, SiftPoint *d_Sift, int fstPts, float subsampling)
{
  int totPts = min(d_PointCounter[0], d_MaxNumPoints);
  if (totPts>fstPts) {
    dim3 blocks0(totPts - fstPts);
    dim3 threads0(128);
    ComputeOrientations<<<blocks0, threads0>>>(texObj, d_Sift, fstPts);
    totPts = min(d_PointCounter[0], d_MaxNumPoints);
    dim3 blocks1(totPts - fstPts);
    dim3 threads1(16, 8);
    ExtractSiftDescriptors<<<blocks1, threads1>>>(texObj, d_Sift, fstPts, subsampling);
  }
}
#endif


///////////////////////////////////////////////////////////////////////////////
// Subtract two images (multi-scale version)
///////////////////////////////////////////////////////////////////////////////

__global__ void FindPointsMulti(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave)
{
#define MEMWID (MINMAX_W + 2)
    __shared__ float ymin1[MEMWID], ymin2[MEMWID], ymin3[MEMWID];
    __shared__ float ymax1[MEMWID], ymax2[MEMWID], ymax3[MEMWID];
    __shared__ unsigned int cnt;
    __shared__ unsigned short points[96];

    int tx = threadIdx.x;
    int block = blockIdx.x/NUM_SCALES;
    int scale = blockIdx.x - NUM_SCALES*block;
    int minx = block*MINMAX_W;
    int maxx = min(minx + MINMAX_W, width);
    int xpos = minx + tx;
    int size = pitch*height;
    int ptr = size*scale + max(min(xpos-1, width-1), 0);

    if (tx==0)
        cnt = 0;
    __syncthreads();

    int yloops = min(height - MINMAX_H*blockIdx.y, MINMAX_H);
    for (int y=0;y<yloops;y++) {

        int ypos = MINMAX_H*blockIdx.y + y;
        int yptr1 = ptr + ypos*pitch;
        float maxv = fabs(d_Data0[yptr1 + 1*size]);
        maxv = fmaxf(maxv, ShiftDown(maxv, 16));
        maxv = fmaxf(maxv, ShiftDown(maxv, 8));
        maxv = fmaxf(maxv, ShiftDown(maxv, 4));
        maxv = fmaxf(maxv, ShiftDown(maxv, 2));
        maxv = fmaxf(maxv, ShiftDown(maxv, 1));
        ymax2[tx] = maxv;
        __syncthreads();
        if (fmaxf(ymax2[0], ymax2[32])<=thresh)
            continue;
        int yptr0 = ptr + max(0,ypos-1)*pitch;
        int yptr2 = ptr + min(height-1,ypos+1)*pitch;
        {
            float d10 = d_Data0[yptr0];
            float d11 = d_Data0[yptr1];
            float d12 = d_Data0[yptr2];
            ymin1[tx] = fminf(fminf(d10, d11), d12);
            ymax1[tx] = fmaxf(fmaxf(d10, d11), d12);
        }
        {
            float d30 = d_Data0[yptr0 + 2*size];
            float d31 = d_Data0[yptr1 + 2*size];
            float d32 = d_Data0[yptr2 + 2*size];
            ymin3[tx] = fminf(fminf(d30, d31), d32);
            ymax3[tx] = fmaxf(fmaxf(d30, d31), d32);
        }
        float d20 = d_Data0[yptr0 + 1*size];
        float d21 = d_Data0[yptr1 + 1*size];
        float d22 = d_Data0[yptr2 + 1*size];
        ymin2[tx] = fminf(fminf(ymin1[tx], fminf(fminf(d20, d21), d22)), ymin3[tx]);
        ymax2[tx] = fmaxf(fmaxf(ymax1[tx], fmaxf(fmaxf(d20, d21), d22)), ymax3[tx]);
        __syncthreads();
        if (tx>0 && tx<MINMAX_W+1 && xpos<=maxx) {
            if (d21<-thresh) {
                float minv = fminf(fminf(fminf(ymin2[tx-1], ymin2[tx+1]), ymin1[tx]), ymin3[tx]);
                minv = fminf(fminf(minv, d20), d22);
                if (d21<minv) {
                    int pos = atomicInc(&cnt, 31);
                    points[3*pos+0] = xpos - 1;
                    points[3*pos+1] = ypos;
                    points[3*pos+2] = scale;
                }
            }
            if (d21>thresh) {
                float maxv = fmaxf(fmaxf(fmaxf(ymax2[tx-1], ymax2[tx+1]), ymax1[tx]), ymax3[tx]);
                maxv = fmaxf(fmaxf(maxv, d20), d22);
                if (d21>maxv) {
                    int pos = atomicInc(&cnt, 31);
                    points[3*pos+0] = xpos - 1;
                    points[3*pos+1] = ypos;
                    points[3*pos+2] = scale;
                }
            }
        }
        __syncthreads();
    }
    if (tx==0)
        atomicMax(&d_PointCounter[2*octave+0], d_PointCounter[2*octave-1]);
    if (tx<cnt) {
        int xpos = points[3*tx+0];
        int ypos = points[3*tx+1];
        int scale = points[3*tx+2];
        int ptr = xpos + (ypos + (scale+1)*height)*pitch;
        float val = d_Data0[ptr];
        float *data1 = &d_Data0[ptr];
        float dxx = 2.0f*val - data1[-1] - data1[1];
        float dyy = 2.0f*val - data1[-pitch] - data1[pitch];
        float dxy = 0.25f*(data1[+pitch+1] + data1[-pitch-1] - data1[-pitch+1] - data1[+pitch-1]);
        float tra = dxx + dyy;
        float det = dxx*dyy - dxy*dxy;
        if (tra*tra<edgeLimit*det) {
            float edge = __fdividef(tra*tra, det);
            float dx = 0.5f*(data1[1] - data1[-1]);
            float dy = 0.5f*(data1[pitch] - data1[-pitch]);
            float *data0 = d_Data0 + ptr - height*pitch;
            float *data2 = d_Data0 + ptr + height*pitch;
            float ds = 0.5f*(data0[0] - data2[0]);
            float dss = 2.0f*val - data2[0] - data0[0];
            float dxs = 0.25f*(data2[1] + data0[-1] - data0[1] - data2[-1]);
            float dys = 0.25f*(data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
            float idxx = dyy*dss - dys*dys;
            float idxy = dys*dxs - dxy*dss;
            float idxs = dxy*dys - dyy*dxs;
            float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
            float idyy = dxx*dss - dxs*dxs;
            float idys = dxy*dxs - dxx*dys;
            float idss = dxx*dyy - dxy*dxy;
            float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
            float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
            float pds = idet*(idxs*dx + idys*dy + idss*ds);
            if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
                pdx = __fdividef(dx, dxx);
                pdy = __fdividef(dy, dyy);
                pds = __fdividef(ds, dss);
            }
            float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
            int maxPts = d_MaxNumPoints;
            float sc = powf(2.0f, (float)scale/NUM_SCALES) * exp2f(pds*factor);
            if (sc>=lowestScale) {
                unsigned int idx = atomicInc(&d_PointCounter[2*octave+0], 0x7fffffff);
                //d_PointCounter[2*octave+1] = d_PointCounter[2*octave+0];
                //printf("Extract %d %d %d %d\n", octave, d_PointCounter[2*octave-1], d_PointCounter[2*octave+0], d_PointCounter[2*octave+1]);
                idx = (idx>=maxPts ? maxPts-1 : idx);
                d_Sift[idx].xpos = xpos + pdx;
                d_Sift[idx].ypos = ypos + pdy;
                d_Sift[idx].scale = sc;
                d_Sift[idx].sharpness = val + dval;
                d_Sift[idx].edgeness = edge;
                d_Sift[idx].subsampling = subsampling;
            }
        }
    }
}


__global__ void LaplaceMultiTex(cudaTextureObject_t texObj, float *d_Result, int width, int pitch, int height, int octave)
{
    __shared__ float data1[(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S];
    __shared__ float data2[LAPLACE_W*LAPLACE_S];
    const int tx = threadIdx.x;
    const int xp = blockIdx.x*LAPLACE_W + tx;
    const int yp = blockIdx.y;
    const int scale = threadIdx.y;
    float *kernel = d_LaplaceKernel + octave*12*16 + scale*16;
    float *sdata1 = data1 + (LAPLACE_W + 2*LAPLACE_R)*scale;
    float x = xp-3.5;
    float y = yp+0.5;
    sdata1[tx] = kernel[4]*tex2D<float>(texObj, x, y) +
                 kernel[3]*(tex2D<float>(texObj, x, y-1.0) + tex2D<float>(texObj, x, y+1.0)) +
                 kernel[2]*(tex2D<float>(texObj, x, y-2.0) + tex2D<float>(texObj, x, y+2.0)) +
                 kernel[1]*(tex2D<float>(texObj, x, y-3.0) + tex2D<float>(texObj, x, y+3.0)) +
                 kernel[0]*(tex2D<float>(texObj, x, y-4.0) + tex2D<float>(texObj, x, y+4.0));
    __syncthreads();
    float *sdata2 = data2 + LAPLACE_W*scale;
    if (tx<LAPLACE_W) {
        sdata2[tx] = kernel[4]*sdata1[tx+4] +
                     kernel[3]*(sdata1[tx+3] + sdata1[tx+5]) +
                     kernel[2]*(sdata1[tx+2] + sdata1[tx+6]) +
                     kernel[1]*(sdata1[tx+1] + sdata1[tx+7]) +
                     kernel[0]*(sdata1[tx+0] + sdata1[tx+8]);
    }
    __syncthreads();
    if (tx<LAPLACE_W && scale<LAPLACE_S-1 && xp<width)
        d_Result[scale*height*pitch + yp*pitch + xp] = sdata2[tx] - sdata2[tx+LAPLACE_W];
}


__global__ void LaplaceMultiMemNew(float *d_Image, float *d_Result, int width, int pitch, int height, int octave)
{
    __shared__ float data1[(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S];
    __shared__ float data2[LAPLACE_W*LAPLACE_S];
    const int tx = threadIdx.x;
    const int xp = blockIdx.x*LAPLACE_W + tx;
    const int yp = 4*blockIdx.y;
    const int scale = threadIdx.y;
    float *kernel = d_LaplaceKernel + octave*12*16 + scale*16;
    float *sdata1 = data1 + (LAPLACE_W + 2*LAPLACE_R)*scale;
    float *data = d_Image + max(min(xp - 4, width-1), 0);
    int h = height-1;
    float temp[12];
    for (int i=0;i<4;i++)
        temp[i] = data[max(0, min(yp+i-4, h))*pitch];
    for (int i=4;i<12;i++)
        temp[i] = data[min(yp+i-4, h)*pitch];
    __syncthreads();
    for (int j=0;j<4;j++) {
        sdata1[tx] = kernel[4]*temp[4+j] +
                     kernel[3]*(temp[3+j] + temp[5+j]) + kernel[2]*(temp[2+j] + temp[6+j]) +
                     kernel[1]*(temp[1+j] + temp[7+j]) + kernel[0]*(temp[0+j] + temp[8+j]);
        __syncthreads();
        float *sdata2 = data2 + LAPLACE_W*scale;
        if (tx<LAPLACE_W) {
            sdata2[tx] = kernel[4]*sdata1[tx+4] +
                         kernel[3]*(sdata1[tx+3] + sdata1[tx+5]) + kernel[2]*(sdata1[tx+2] + sdata1[tx+6]) +
                         kernel[1]*(sdata1[tx+1] + sdata1[tx+7]) + kernel[0]*(sdata1[tx+0] + sdata1[tx+8]);
        }
        __syncthreads();
        if (tx<LAPLACE_W && scale<LAPLACE_S-1 && xp<width && (yp+j)<height)
            d_Result[scale*height*pitch + (yp+j)*pitch + xp] = sdata2[tx] - sdata2[tx+LAPLACE_W];
        __syncthreads();
    }
}

__global__ void LaplaceMultiMem(float *d_Image, float *d_Result, int width, int pitch, int height, int octave)
{
    __shared__ float data1[(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S];
    __shared__ float data2[LAPLACE_W*LAPLACE_S];
    const int tx = threadIdx.x;
    const int xp = blockIdx.x*LAPLACE_W + tx;
    const int yp = blockIdx.y;
    const int scale = threadIdx.y;
    float *kernel = d_LaplaceKernel + octave*12*16 + scale*16;
    float *sdata1 = data1 + (LAPLACE_W + 2*LAPLACE_R)*scale;
    float *data = d_Image + max(min(xp - 4, width-1), 0);
    int h = height-1;
    sdata1[tx] = kernel[4]*data[min(yp, h)*pitch] +
                 kernel[3]*(data[max(0, min(yp-1, h))*pitch] + data[min(yp+1, h)*pitch]) +
                 kernel[2]*(data[max(0, min(yp-2, h))*pitch] + data[min(yp+2, h)*pitch]) +
                 kernel[1]*(data[max(0, min(yp-3, h))*pitch] + data[min(yp+3, h)*pitch]) +
                 kernel[0]*(data[max(0, min(yp-4, h))*pitch] + data[min(yp+4, h)*pitch]);
    __syncthreads();
    float *sdata2 = data2 + LAPLACE_W*scale;
    if (tx<LAPLACE_W) {
        sdata2[tx] = kernel[4]*sdata1[tx+4] +
                     kernel[3]*(sdata1[tx+3] + sdata1[tx+5]) + kernel[2]*(sdata1[tx+2] + sdata1[tx+6]) +
                     kernel[1]*(sdata1[tx+1] + sdata1[tx+7]) + kernel[0]*(sdata1[tx+0] + sdata1[tx+8]);
    }
    __syncthreads();
    if (tx<LAPLACE_W && scale<LAPLACE_S-1 && xp<width)
        d_Result[scale*height*pitch + yp*pitch + xp] = sdata2[tx] - sdata2[tx+LAPLACE_W];
}

__global__ void LowPass(float *d_Image, float *d_Result, int width, int pitch, int height)
{
    __shared__ float buffer[(LOWPASS_W + 2*LOWPASS_R)*LOWPASS_H];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int xp = blockIdx.x*LOWPASS_W + tx;
    const int yp = blockIdx.y*LOWPASS_H + ty;
    float *kernel = d_LowPassKernel;
    float *data = d_Image + max(min(xp - 4, width-1), 0);
    float *buff = buffer + ty*(LOWPASS_W + 2*LOWPASS_R);
    int h = height-1;
    if (yp<height)
        buff[tx] = kernel[4]*data[min(yp, h)*pitch] +
                   kernel[3]*(data[max(0, min(yp-1, h))*pitch] + data[min(yp+1, h)*pitch]) +
                   kernel[2]*(data[max(0, min(yp-2, h))*pitch] + data[min(yp+2, h)*pitch]) +
                   kernel[1]*(data[max(0, min(yp-3, h))*pitch] + data[min(yp+3, h)*pitch]) +
                   kernel[0]*(data[max(0, min(yp-4, h))*pitch] + data[min(yp+4, h)*pitch]);
    __syncthreads();
    if (tx<LOWPASS_W && xp<width && yp<height) {
        d_Result[yp*pitch + xp] = kernel[4]*buff[tx+4] +
                                  kernel[3]*(buff[tx+3] + buff[tx+5]) + kernel[2]*(buff[tx+2] + buff[tx+6]) +
                                  kernel[1]*(buff[tx+1] + buff[tx+7]) + kernel[0]*(buff[tx+0] + buff[tx+8]);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////


void InitCuda(int devNum)
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (!nDevices) {
        std::cerr << "No CUDA devices available" << std::endl;
        return;
    }
    devNum = std::min(nDevices-1, devNum);
    deviceInit(devNum);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devNum);
    printf("Device Number: %d\n", devNum);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1000);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp)
{
    TimerGPU timer(0);
    const int nd = NUM_SCALES + 3;
    int w = width*(scaleUp ? 2 : 1);
    int h = height*(scaleUp ? 2 : 1);
    int p = iAlignUp(w, 128);
    int size = h*p;                 // image sizes
    int sizeTmp = nd*h*p;           // laplace buffer sizes
    for (int i=0;i<numOctaves;i++) {
        w /= 2;
        h /= 2;
        int p = iAlignUp(w, 128);
        size += h*p;
        sizeTmp += nd*h*p;
    }
    float *memoryTmp = NULL;
    size_t pitch;
    size += sizeTmp;
    safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
#ifdef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
  printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
    return memoryTmp;
}

void FreeSiftTempMemory(float *memoryTmp)
{
    if (memoryTmp)
        safeCall(cudaFree(memoryTmp));
}

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float *tempMemory)
{
    TimerGPU timer(0);
    unsigned int *d_PointCounterAddr;
    safeCall(cudaGetSymbolAddress((void**)&d_PointCounterAddr, d_PointCounter));
    safeCall(cudaMemset(d_PointCounterAddr, 0, (8*2+1)*sizeof(int)));
    safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)));

    const int nd = NUM_SCALES + 3;
    int w = img.width*(scaleUp ? 2 : 1);
    int h = img.height*(scaleUp ? 2 : 1);
    int p = iAlignUp(w, 128);
    int width = w, height = h;
    int size = h*p;                 // image sizes
    int sizeTmp = nd*h*p;           // laplace buffer sizes
    for (int i=0;i<numOctaves;i++) {
        w /= 2;
        h /= 2;
        int p = iAlignUp(w, 128);
        size += h*p;
        sizeTmp += nd*h*p;
    }
    float *memoryTmp = tempMemory;
    size += sizeTmp;
    if (!tempMemory) {
        size_t pitch;
        safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
#ifdef VERBOSE
        printf("Allocated memory size: %d bytes\n", size);
    printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
    }
    float *memorySub = memoryTmp + sizeTmp;

    CudaImage lowImg;
    lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);
    if (!scaleUp) {
        LowPass(lowImg, img, max(initBlur, 0.001f));
        TimerGPU timer1(0);
        float kernel[8*12*16];
        PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
        safeCall(cudaMemcpyToSymbol(d_LaplaceKernel, kernel, 8*12*16*sizeof(float)));
        int octave = ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
        safeCall(cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2*octave], sizeof(int), cudaMemcpyDeviceToHost));
        siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
//        printf("SIFT extraction time =        %.2f ms\n", timer1.read());
    } else {
        CudaImage upImg;
        upImg.Allocate(width, height, iAlignUp(width, 128), false, memoryTmp);
        TimerGPU timer1(0);
        ScaleUp(upImg, img);
        LowPass(lowImg, upImg, max(initBlur, 0.001f));
        float kernel[8*12*16];
        PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
        safeCall(cudaMemcpyToSymbol(d_LaplaceKernel, kernel, 8*12*16*sizeof(float)));
        int octave = ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale*2.0f, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
        safeCall(cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2*octave], sizeof(int), cudaMemcpyDeviceToHost));
        siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
        RescalePositions(siftData, 0.5f);
//        printf("SIFT extraction time =        %.2f ms\n", timer1.read());
    }

    if (!tempMemory)
        safeCall(cudaFree(memoryTmp));
#ifdef MANAGEDMEM
    safeCall(cudaDeviceSynchronize());
#else
    if (siftData.h_data)
        safeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint)*siftData.numPts, cudaMemcpyDeviceToHost));
#endif
    double totTime = timer.read();
//    printf("Incl prefiltering & memcpy =  %.2f ms %d\n\n", totTime, siftData.numPts);
}

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub)
{
#ifdef VERBOSE
    TimerGPU timer(0);
#endif
    int w = img.width;
    int h = img.height;
    if (numOctaves>1) {
        CudaImage subImg;
        int p = iAlignUp(w/2, 128);
        subImg.Allocate(w/2, h/2, p, false, memorySub);
        ScaleDown(subImg, img, 0.5f);
        float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
        ExtractSiftLoop(siftData, subImg, numOctaves-1, totInitBlur, thresh, lowestScale, subsampling*2.0f, memoryTmp, memorySub + (h/2)*p);
    }
    if (lowestScale<subsampling*2.0f)
        ExtractSiftOctave(siftData, img, numOctaves, thresh, lowestScale, subsampling, memoryTmp);
#ifdef VERBOSE
    double totTime = timer.read();
  printf("ExtractSift time total =      %.2f ms\n\n", totTime);
#endif
    if (lowestScale<subsampling*2.0f)
        return numOctaves;
    return 0;
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling, float *memoryTmp)
//void ExtractSiftOctave(SiftData &siftData, CudaImage &img, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp)
{
    const int nd = NUM_SCALES + 3;
#ifdef VERBOSE
    TimerGPU timer0;
#endif
    CudaImage diffImg[nd];
    int w = img.width;
    int h = img.height;
    int p = iAlignUp(w, 128);
    for (int i=0;i<nd-1;i++)
        diffImg[i].Allocate(w, h, p, false, memoryTmp + i*p*h);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = img.d_data;
    resDesc.res.pitch2D.width = img.width;
    resDesc.res.pitch2D.height = img.height;
    resDesc.res.pitch2D.pitchInBytes = img.pitch*sizeof(float);
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

#ifdef VERBOSE
    TimerGPU timer1;
#endif
    float baseBlur = pow(2.0f, -1.0f/NUM_SCALES);
    float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
    LaplaceMulti(texObj, img, diffImg, octave);
    FindPointsMulti(diffImg, siftData, thresh, 10.0f, 1.0f/NUM_SCALES, lowestScale/subsampling, subsampling, octave);
#ifdef VERBOSE
    double gpuTimeDoG = timer1.read();
  TimerGPU timer4;
#endif
    ComputeOrientations(texObj, siftData, octave);
    ExtractSiftDescriptors(texObj, siftData, subsampling, octave);

    safeCall(cudaDestroyTextureObject(texObj));
#ifdef VERBOSE
    double gpuTimeSift = timer4.read();
  double totTime = timer0.read();
  printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime-gpuTimeDoG-gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
  safeCall(cudaMemcpyFromSymbol(&totPts, &d_PointCounter[2], sizeof(int)));
  int totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts>0)
    printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n", gpuTimeDoG/NUM_SCALES, gpuTimeSift/(totPts-fstPts), totPts-fstPts);
#endif
}

void InitSiftData(SiftData &data, int num, bool host, bool dev)
{
    data.numPts = 0;
    data.maxPts = num;
    int sz = sizeof(SiftPoint)*num;
#ifdef MANAGEDMEM
    safeCall(cudaMallocManaged((void **)&data.m_data, sz));
#else
    data.h_data = NULL;
    if (host)
        data.h_data = (SiftPoint *)malloc(sz);
    data.d_data = NULL;
    if (dev)
        safeCall(cudaMalloc((void **)&data.d_data, sz));
#endif
}

void FreeSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
    safeCall(cudaFree(data.m_data));
#else
    if (data.d_data!=NULL)
        safeCall(cudaFree(data.d_data));
    data.d_data = NULL;
    if (data.h_data!=NULL)
        free(data.h_data);
#endif
    data.numPts = 0;
    data.maxPts = 0;
}

void PrintSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
    SiftPoint *h_data = data.m_data;
#else
    SiftPoint *h_data = data.h_data;
    if (data.h_data==NULL) {
        h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*data.maxPts);
        safeCall(cudaMemcpy(h_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToHost));
        data.h_data = h_data;
    }
#endif
    for (int i=0;i<data.numPts;i++) {
        printf("xpos         = %.2f\n", h_data[i].xpos);
        printf("ypos         = %.2f\n", h_data[i].ypos);
        printf("scale        = %.2f\n", h_data[i].scale);
        printf("sharpness    = %.2f\n", h_data[i].sharpness);
        printf("edgeness     = %.2f\n", h_data[i].edgeness);
        printf("orientation  = %.2f\n", h_data[i].orientation);
        printf("score        = %.2f\n", h_data[i].score);
        float *siftData = (float*)&h_data[i].data;
        for (int j=0;j<8;j++) {
            if (j==0)
                printf("data = ");
            else
                printf("       ");
            for (int k=0;k<16;k++)
                if (siftData[j+8*k]<0.05)
                    printf(" .   ");
                else
                    printf("%.2f ", siftData[j+8*k]);
            printf("\n");
        }
    }
    printf("Number of available points: %d\n", data.numPts);
    printf("Number of allocated points: %d\n", data.maxPts);
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double ScaleDown(CudaImage &res, CudaImage &src, float variance)
{
    static float oldVariance = -1.0f;
    if (res.d_data==NULL || src.d_data==NULL) {
        printf("ScaleDown: missing data\n");
        return 0.0;
    }
    if (oldVariance!=variance) {
        float h_Kernel[5];
        float kernelSum = 0.0f;
        for (int j=0;j<5;j++) {
            h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);
            kernelSum += h_Kernel[j];
        }
        for (int j=0;j<5;j++)
            h_Kernel[j] /= kernelSum;
        safeCall(cudaMemcpyToSymbol(d_ScaleDownKernel, h_Kernel, 5*sizeof(float)));
        oldVariance = variance;
    }
    dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
    dim3 threads(SCALEDOWN_W + 4);
    ScaleDown<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
    checkMsg("ScaleDown() execution failed\n");
    return 0.0;
}

double ScaleUp(CudaImage &res, CudaImage &src)
{
    if (res.d_data==NULL || src.d_data==NULL) {
        printf("ScaleUp: missing data\n");
        return 0.0;
    }
    dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
    dim3 threads(SCALEUP_W, SCALEUP_H);
    ScaleUp<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
    checkMsg("ScaleUp() execution failed\n");
    return 0.0;
}

double ComputeOrientations(cudaTextureObject_t texObj, SiftData &siftData, int octave)
{
    dim3 blocks(256);
    dim3 threads(128);
#ifdef MANAGEDMEM
    ComputeOrientationsCONST<<<blocks, threads>>>(texObj, siftData.m_data, octave);
#else
    ComputeOrientationsCONST<<<blocks, threads>>>(texObj, siftData.d_data, octave);
#endif
    checkMsg("ComputeOrientations() execution failed\n");
    return 0.0;
}

double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData,float subsampling, int octave)
{
    dim3 blocks(256);
    dim3 threads(16, 8);
#ifdef MANAGEDMEM
    ExtractSiftDescriptorsCONST<<<blocks, threads>>>(texObj, siftData.m_data, subsampling, octave);
#else
    ExtractSiftDescriptorsCONST<<<blocks, threads>>>(texObj, siftData.d_data, subsampling, octave);
#endif
    checkMsg("ExtractSiftDescriptors() execution failed\n");
    return 0.0;
}

#if 0
double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, float subsampling)
{
#ifdef MANAGEDMEM
  OrientAndExtract<<<1,1>>>(texObj, siftData.m_data, fstPts, subsampling);
#else
  OrientAndExtract<<<1,1>>>(texObj, siftData.d_data, fstPts, subsampling);
#endif
  checkMsg("OrientAndExtract() execution failed\n");
  return 0.0;
}
#endif

double RescalePositions(SiftData &siftData, float scale)
{
    dim3 blocks(iDivUp(siftData.numPts, 64));
    dim3 threads(64);
    RescalePositions<<<blocks, threads>>>(siftData.d_data, siftData.numPts, scale);
    checkMsg("RescapePositions() execution failed\n");
    return 0.0;
}

double LowPass(CudaImage &res, CudaImage &src, float scale)
{
    float kernel[2*LOWPASS_R+1];
    static float oldScale = -1.0f;
    if (scale!=oldScale) {
        float kernelSum = 0.0f;
        float ivar2 = 1.0f/(2.0f*scale*scale);
        for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) {
            kernel[j+LOWPASS_R] = (float)expf(-(double)j*j*ivar2);
            kernelSum += kernel[j+LOWPASS_R];
        }
        for (int j=-LOWPASS_R;j<=LOWPASS_R;j++)
            kernel[j+LOWPASS_R] /= kernelSum;
        safeCall(cudaMemcpyToSymbol(d_LowPassKernel, kernel, (2*LOWPASS_R+1)*sizeof(float)));
        oldScale = scale;
    }
    int width = res.width;
    int pitch = res.pitch;
    int height = res.height;
    dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
    dim3 threads(LOWPASS_W+2*LOWPASS_R, LOWPASS_H);
    LowPass<<<blocks, threads>>>(src.d_data, res.d_data, width, pitch, height);
    checkMsg("LowPass() execution failed\n");
    return 0.0;
}

//==================== Multi-scale functions ===================//

void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel)
{
    if (numOctaves>1) {
        float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
        PrepareLaplaceKernels(numOctaves-1, totInitBlur, kernel);
    }
    float scale = pow(2.0f, -1.0f/NUM_SCALES);
    float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
    for (int i=0;i<NUM_SCALES+3;i++) {
        float kernelSum = 0.0f;
        float var = scale*scale - initBlur*initBlur;
        for (int j=-LAPLACE_R;j<=LAPLACE_R;j++) {
            kernel[numOctaves*12*16 + 16*i+j+LAPLACE_R] = (float)expf(-(double)j*j/2.0/var);
            kernelSum += kernel[numOctaves*12*16 + 16*i+j+LAPLACE_R];
        }
        for (int j=-LAPLACE_R;j<=LAPLACE_R;j++)
            kernel[numOctaves*12*16 + 16*i+j+LAPLACE_R] /= kernelSum;
        scale *= diffScale;
    }
}

double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave)
{
    int width = results[0].width;
    int pitch = results[0].pitch;
    int height = results[0].height;
    dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
#if 1
    dim3 blocks(iDivUp(width, LAPLACE_W), iDivUp(height, 4));
    LaplaceMultiMemNew<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#else
    dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiTex<<<blocks, threads>>>(texObj, results[0].d_data, width, pitch, height, octave);
#endif
    checkMsg("LaplaceMulti() execution failed\n");
    return 0.0;
}

double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave)
{
    if (sources->d_data==NULL) {
        printf("FindPointsMulti: missing data\n");
        return 0.0;
    }
    int w = sources->width;
    int p = sources->pitch;
    int h = sources->height;
    dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
    dim3 threads(MINMAX_W + 2);
#ifdef MANAGEDMEM
    FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.m_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#else
    FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#endif
    checkMsg("FindPointsMulti() execution failed\n");
    return 0.0;
}


////////////////////////////////////////////////////////////////////////////////////////////////


SiftData::SiftData(int num)
        : h_data(NULL),
          d_data(NULL)
{
    InitSiftData(*this, num, true, true);
}

SiftData::SiftData(const SiftData &data)
        : h_data(NULL),
          d_data(NULL)
{
    InitSiftData(*this, data.maxPts, true, true);
    numPts = data.numPts;
    if (h_data != NULL && data.h_data != NULL) std::memcpy(h_data, data.h_data, numPts*sizeof(SiftPoint));
    if (d_data != NULL && data.d_data != NULL) safeCall(cudaMemcpy(d_data, data.d_data, numPts*sizeof(SiftPoint), cudaMemcpyDeviceToDevice));
}

SiftData::~SiftData() {
    FreeSiftData(*this);
}

SiftData& SiftData::operator=(const SiftData &data) {
    if (this != &data) {
        InitSiftData(*this, data.maxPts, true, true);
        numPts = data.numPts;
        if (h_data != NULL && data.h_data != NULL) std::memcpy(h_data, data.h_data, numPts*sizeof(SiftPoint));
        if (d_data != NULL && data.d_data != NULL) safeCall(cudaMemcpy(d_data, data.d_data, numPts*sizeof(SiftPoint), cudaMemcpyDeviceToDevice));
    }
    return *this;
}

void SiftData::resize(size_t new_size) {
    if (new_size <= (size_t)maxPts) {
        numPts = (int)new_size;
        return;
    }
    reserve(new_size);  // Conservative?
    numPts = (int)new_size;
}

void SiftData::reserve(size_t new_capacity) {
    if (new_capacity <= (size_t)maxPts) return;

    size_t mem_sz = new_capacity*sizeof(SiftPoint);

    if (h_data != NULL) {
        SiftPoint * h_data_old = h_data;
        h_data = (SiftPoint *)malloc(mem_sz);
        std::memcpy(h_data, h_data_old, numPts*sizeof(SiftPoint));
        free(h_data_old);
    } else {
        h_data = (SiftPoint *)malloc(mem_sz);
    }

    if (d_data != NULL) {
        SiftPoint * d_data_old = d_data;
        safeCall(cudaMalloc((void **)&d_data, mem_sz));
        safeCall(cudaMemcpy(d_data, d_data_old, numPts*sizeof(SiftPoint), cudaMemcpyDeviceToDevice));
        safeCall(cudaFree(d_data_old));
    } else {
        safeCall(cudaMalloc((void **)&d_data, mem_sz));
    }

    maxPts = (int)new_capacity;
}

void SiftData::freeBuffers() {
    FreeSiftData(*this);
}

SiftData& SiftData::append(const SiftData &data) {
    int new_sz = numPts + data.numPts;
    reserve((size_t)(new_sz));
    size_t cp_sz = data.numPts*sizeof(SiftPoint);
    if (h_data != NULL && data.h_data != NULL) std::memcpy(&h_data[numPts], data.h_data, cp_sz);
    if (d_data != NULL && data.d_data != NULL) safeCall(cudaMemcpy(&d_data[numPts], data.d_data, cp_sz, cudaMemcpyDeviceToDevice));
    numPts = new_sz;
    return *this;
}

void SiftData::syncHostToDevice() {
    if (h_data != NULL && d_data != NULL) safeCall(cudaMemcpy(d_data, h_data, numPts*sizeof(SiftPoint), cudaMemcpyHostToDevice));
}

void SiftData::syncDeviceToHost() {
    if (h_data != NULL && d_data != NULL) safeCall(cudaMemcpy(h_data, d_data, numPts*sizeof(SiftPoint), cudaMemcpyDeviceToHost));
}
