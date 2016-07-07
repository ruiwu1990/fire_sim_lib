#include "kernel_IMT.h"
/////////////////////////////////////////////////////////////////////////////
//                          Iterative Minimal Time
/////////////////////////////////////////////////////////////////////////////
__global__ void ItMinTime(int* ignTimeIn, int* ignTimeOut, int* ignTimeStep,
                          float* rothData, int* times, float* L_n, bool* check,
                          int size, int rowSize, int colSize){
   /* neighbor's address*/     /* N  NE   E  SE   S  SW   W  NW  NNW NNE NEE SEE SSE SSW SWW NWW*/
   int nCol[16] =        {  0,  1,  1,  1,  0, -1, -1, -1, -1, 1, 2, 2, 1, -1, -2, -2};
   int nRow[16] =        {  1,  1,  0, -1, -1, -1,  0,  1, 2, 2, 1, -1, -2, -2, -1, 1};
   float angles[16] =        {  90,  45,  0, -45, -90, -115,  180,  115, 63.4349, 26.5651, -26.5651,
                                -63.4349, -116.5651, -153.4349, 153.4349, 116.5651};
   // printf("Iterative Minimal Time\n");
   float ignCell = 0;
   float ignCellNew = 0;
   float ignTimeMin = INF;

   int cell = blockIdx.x * blockDim.x + threadIdx.x;
   int ncell, nrow, ncol, row, col, rothCell;
   float ignTimeNew, ROS, ROS_Update;

   while(cell < size){
//      row = cell / rowSize;
//      col = cell - rowSize*row;
      row = cell / colSize;
      col = cell % colSize;
      // printf("%d ", cell);

      // Do nothing if converged
      if(check[cell] == true){
         cell += blockDim.x * gridDim.x;
         // atomicAdd(&end, 1);
         // printf("if_statement_1\n");
         continue;
      }

      // Check for simulation completion
      ignCell = ignTimeIn[cell];
      ignCellNew = ignTimeOut[cell];
      // Convergence Test
      if(fabs(ignCell - ignCellNew) < 2 && ignCell != INF
         && ignCellNew != INF && check[cell] != true){
         check[cell] = true;
         cell += blockDim.x * gridDim.x;
         // atomicAdd(&end, 1);
         continue;
      }
      if(ignCell > 0){
         // ignTimeMin = INF;
         // printf("ignCell > 0\n");
         ignTimeMin = INF;
         // Loop through neighbors
         for(int n = 0; n < 16; n++){
            // find neighbor cell index
            nrow = row + nRow[n];
            ncol = col + nCol[n];
            if ( nrow<0 || nrow>= rowSize || ncol<0 || ncol>=  colSize ){
               continue;
            }
            ncell = ncol + nrow*colSize;

            ROS = rothData[ncell * 16 + n];
            ignTimeNew = ignTimeIn[ncell] + L_n[n] / ROS;// * 100;

            ignTimeMin = ignTimeNew*(ignTimeNew < ignTimeMin) + ignTimeMin*(ignTimeNew >= ignTimeMin);
            ROS_Update = ROS*(ignTimeNew < ignTimeMin) + rothData[cell]*(ignTimeNew >= ignTimeMin);
            // printf("ignTimeNewMin: %f \n", ign_time_new_);
         }
         // ignTimeStep[cell] = (int)ignTimeMin; // atomic min here?
         if(ignTimeMin >0){
            ignTimeOut[cell] = (int)ignTimeMin;
//            printf("%f", ignTimeMin)
            rothData[cell] = ROS_Update;
         }
         // atomicMin(&ignTimeOut[cell], (int)ignTimeMin);
      }
      cell += blockDim.x * gridDim.x;
   }
   // printf("Testing IMT Kernel\n");
   if(blockIdx.x * blockDim.x + threadIdx.x == 0)
      end = 0;
}

/////////////////////////////////////////////////////////////////////////////
//                             Copy Kernel (IMT)
/////////////////////////////////////////////////////////////////////////////
//__global__ void copyKernelIMT(int* input, int* output, bool* check, int size){
//   // copy from output to input
//   int cell = blockIdx.x * blockDim.x + threadIdx.x;
//   // printf("%d ", cell);
//   // end = 0;
//
//   while(cell < size){
//      // input[cell] = output[cell];
//      if(check[cell] == true){
//         // printf("true\n");
//         atomicAdd(&end, 1);
//      }
//      cell += blockDim.x * gridDim.x;
//      // printf("%d ", end);
//   }
//} 