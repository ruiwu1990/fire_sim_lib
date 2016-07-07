#include <curand_mtgp32_kernel.h>
#include "kernel_BD.h"

//const int INF = 32767;
__device__ int end;
//__device__ int syncCounter;

/////////////////////////////////////////////////////////////////////////////
//                            Burning Distances
/////////////////////////////////////////////////////////////////////////////
__global__ void BurnDist(float* ignTimeIn, float* ignTimeOut,float* burnDist, float* maxspreadrate,
                         float* currentspreadrate, float* times,float* L_n, int size,
                         int rowSize, int colSize, float timeStep, float t){
   /* neighbor's address*/     /* N  NE   E  SE   S  SW   W  NW  NNW NNE NEE SEE SSE SSW SWW NWW*/
   int nCol[8] =        {  0,  1,  1,  1,  0, -1, -1, -1};
   int nRow[8] =        {  1,  1,  0, -1, -1, -1,  0,  1};
   int angles[8] =        {  90,  45,  0, -45, -90, -115,  180,  115};
//   printf("T: %f\n",t);
//   printf("%f %f\n", L_n[0], L_n[1]);
   float ignTime, ignTimeN;
   float dist;

   int cell = blockIdx.x * blockDim.x + threadIdx.x;
   int ncell, nrow, ncol, row, col, distCell, rothCell;
   float ROS;

   while(cell < size){
      row = cell / colSize;
      col = cell % colSize;
//       printf("%d %d %d\n", cell, rowSize, col);
      ignTime = ignTimeIn[cell];
      if(ignTime == INF){
         cell += blockDim.x * gridDim.x;
         continue;
      }

      // check neighbors for ignition
      for(int n = 0; n < 8; n++){
         // printf("%d ", n);
         distCell = cell * 8;
         rothCell = cell * 16;
         nrow = row + nRow[n];
         ncol = col + nCol[n];
         if ( nrow<0 || nrow>=rowSize || ncol<0 || ncol>=colSize ) {
            continue;
         }
         ncell = ncol + nrow*colSize;

         // check for already lit
         ignTimeN = ignTimeIn[ncell];
         if(ignTimeN < INF){
            continue;
         }

         // Calc roth values
//         ROS = currentspreadrate[3*cell + 0] * (1.0 - currentspreadrate[3*cell + 1]) /
//               (1.0 - currentspreadrate[3*cell + 1] * cos(currentspreadrate[3*cell + 2] * 3.14159/180.));
//         ROS = currentspreadrate[3*cell + 0] * (1.0 - currentspreadrate[3*cell + 2]) /
//               (1.0 - currentspreadrate[3*cell + 2] * cos(currentspreadrate[3*cell + 1] * 3.14159/180.f - angles[n]));
         ROS = currentspreadrate[rothCell + n];

         // Burn distance
         dist = burnDist[distCell+n];
         dist = dist - ROS*timeStep;
         burnDist[distCell+n] = dist;
//         printf("dist: %f\n", dist);

         // Propagate fire
         if(dist <= 0/* && maxspreadrate[ncell] > 0*/){
//            printf("IN_T: %f\n",t);
            dist *= -1;
            float step_time = dist / ROS;
            step_time += t;
//            float old = atomicExch(&ignTimeOut[ncell], t);
            float old = atomicExch(&ignTimeOut[ncell], step_time);
            if(old < step_time){
               atomicExch(&ignTimeOut[ncell], old);
               currentspreadrate[ncell] = ROS;
            }
         }
      }
      cell += blockDim.x * gridDim.x;
   }
//   printf("End: %d\n", end);
}


/////////////////////////////////////////////////////////////////////////////
//                             Copy Kernel (BD)
/////////////////////////////////////////////////////////////////////////////
__global__ void copyKernelBD(float* input, float* output, int size){
   int cell = blockIdx.x * blockDim.x + threadIdx.x;
   while(cell < size){
      input[cell] = output[cell];
      cell += blockDim.x * gridDim.x;
   }
}