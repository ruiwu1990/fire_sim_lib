#include <curand_mtgp32_kernel.h>
#include "kernel_common.h"


/////////////////////////////////////////////////////////////////////////////
//                             Accelerate
/////////////////////////////////////////////////////////////////////////////
__global__ void Accelerate(float* curspreadrate, float* maxspreadrate, float* acc_const, int size, float time_step){
   int dirs = 16;
   int cell = (blockIdx.x * blockDim.x + threadIdx.x) * dirs;
   float current, ratio, timetomax;
   while(cell < size){
      if(maxspreadrate[cell] <= 0.){
         cell += (blockDim.x * gridDim.x)*dirs;
         continue;
      }
      float acceleration_constant = acc_const[cell];
//      float acceleration_constant = 0.115;
      // Accelerate every cell
      for (int i = 0; i < dirs; i++, cell++) {
         current = !(maxspreadrate[cell+i]<curspreadrate[cell+i])?curspreadrate[cell+i]:maxspreadrate[cell];
         ratio = current / maxspreadrate[cell];
         timetomax = -log(1.0f - ratio) / acceleration_constant;
         curspreadrate[cell] = Clamp(time_step - ratio, 0.0f,1.0f) * (maxspreadrate[cell] - current) + current;
      }
      cell += (blockDim.x * gridDim.x)*dirs;
   }

}



/////////////////////////////////////////////////////////////////////////////
//                             Test Crown Rate
/////////////////////////////////////////////////////////////////////////////
__global__ void TestCrownRate(float* curspreadrate, float* maxspreadrate, float* intensity_modifier, int size,
                              float* I_o, float* RAC){
   int dirs = 16;
   int cell = (blockIdx.x * blockDim.x + threadIdx.x) * dirs;
   int cell_im = (blockIdx.x * blockDim.x + threadIdx.x);
   float I_b, R_max_crown, surface_fuel_consumption, crown_coeff, CFB, crown_rate;
   while(cell < size*dirs){
      if(maxspreadrate[cell] <= 0.f){
         cell += (blockDim.x * gridDim.x)*dirs;
         cell_im += (blockDim.x * gridDim.x);
         continue;
      }

      I_b = curspreadrate[cell] * intensity_modifier[cell_im];
      if(I_b > I_o[cell_im]){
         R_max_crown = 3.34f * maxspreadrate[cell];
         surface_fuel_consumption = I_o[cell_im] * curspreadrate[cell] / I_b;
         crown_coeff = (float) (-log(0.1) / (0.9 * (RAC[cell_im] - surface_fuel_consumption)));
         CFB = (float) (1.0 - exp(-crown_coeff * (curspreadrate[cell] - surface_fuel_consumption)));
         CFB = Clamp(CFB, 0.0, 1.0);
         crown_rate = curspreadrate[cell] + CFB * (R_max_crown - curspreadrate[cell]);
         if(crown_rate >= RAC[cell_im]){
            maxspreadrate[cell] = (crown_rate > maxspreadrate[cell] ? crown_rate : maxspreadrate[cell]);
         }
      }
      cell += (blockDim.x * gridDim.x)*dirs;
      cell_im += (blockDim.x * gridDim.x);
   }
}




/////////////////////////////////////////////////////////////////////////////
//                               Clamp
/////////////////////////////////////////////////////////////////////////////
__device__ float Clamp(float val, float flr, float ceiling){
   if(val >= flr && val <= ceiling){
      return val;
   }
   if(val < flr){
      return flr;
   }
   return ceiling;
}
