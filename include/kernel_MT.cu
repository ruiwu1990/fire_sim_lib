#include "kernel_MT.h"

/////////////////////////////////////////////////////////////////////////////
//                              Minimal Time
/////////////////////////////////////////////////////////////////////////////
__global__ void MinTime(int* ignTime, float* rothData, int* times,
                        float* L_n, int size, int rowSize,
                        int colSize){
   /* neighbor's address*/     /* N  NE   E  SE   S  SW   W  NW  NNW NNE NEE SEE SSE SSW SWW NWW*/
   int nCol[16] =        {  0,  1,  1,  1,  0, -1, -1, -1, -1, 1, 2, 2, 1, -1, -2, -2};
   int nRow[16] =        {  1,  1,  0, -1, -1, -1,  0,  1, 2, 2, 1, -1, -2, -2, -1, 1};
   float angles[16] =        {  90,  45,  0, -45, -90, -115,  180,  115, 63.4349, 26.5651, -26.5651,
                              -63.4349, -116.5651, -153.4349, 153.4349, 116.5651};

   // Calculate ThreadID
   int cell = blockIdx.x * blockDim.x + threadIdx.x;
   int ncell, nrow, ncol, row, col;
   float /*ignCell, ignCellN, time_next_, time_now_,*/ ROS;
   int ignCell, ignCellN, timeNow, timeNext;
   int rothCell = 0;

   timeNow = times[0]; // time_now_ = time_next_
   // printf("%d ", times[1]);
   timeNext = INF;
   // printf("%d ", 5);

   while(cell < size){
      // printf("%d ", cell);
      // time_next_ = INF;
//      row = cell / rowSize;
//      col = cell - rowSize*row;
      row = cell / colSize;
      col = cell % colSize;
      ignCell = ignTime[cell];

      // Do atomic update of TimeNext Var (atomicMin)
      if(timeNext > ignCell && ignCell > timeNow){
         atomicMin(&times[1], ignCell);
         timeNext = ignCell;
      }
      else if(ignCell == timeNow){ // I am on fire now, and will propagate
         // Check through neighbors
         for(int n = 0; n < 16; n++){
            // // Propagate from burning cells
            rothCell = cell * 16;
            nrow = row + nRow[n];
            ncol = col + nCol[n];
            // printf("nrow: %d ncol: %d\n",nrow,ncol);
            if ( nrow<0 || nrow>= rowSize || ncol<0 || ncol>=  colSize ){
               continue;
            }
            ncell = ncol + nrow*colSize;
            ignCellN = ignTime[ncell];

            // If neighbor is unburned in this timestep
            if(ignCellN > timeNow){
               // compute ignition time
//               ROS = rothData[3*cell + 0] * (1.0 - rothData[3*cell + 2]) /
//                     (1.0 - rothData[3*cell + 2] * cos(rothData[3*cell + 1] * 3.14159/180.f - angles[n]));
               ROS = rothData[rothCell + n];
//               if(ROS != 10.761236)
//                  printf("ROS: %f \n", ROS);
               float ignTimeNew = (timeNow + (L_n[n] / ROS));//*100;
               if(ignTimeNew <=0)
                  break;
//               printf("%d\n", ignTimeNew);

               // Update Output TOA Map
               int old = atomicMin(&ignTime[ncell], (int)ignTimeNew);
               if(old != ignTimeNew)
                  rothData[ncell] = ROS;

               // Local time_next_ update
               if((int)ignTimeNew < timeNext && ignTimeNew > 0){ // #thisisnotahacklol
//                  printf("%f, %d\n", ignTimeNew,timeNext);
                  timeNext = (int)ignTimeNew;
//                  if(timeNext == 0)
               }
            }
         }
         // Perform global time_next_ update
         atomicMin(&times[1], timeNext);
      }

      // Do striding
      cell += blockDim.x * gridDim.x;
   }

}

/////////////////////////////////////////////////////////////////////////////
//                             Time Update (MT)
/////////////////////////////////////////////////////////////////////////////
__global__ void TimeKernelMT(int* times){
   times[0] = times[1];
//   if(times[0] == 0)
   times[1] = INF;
}