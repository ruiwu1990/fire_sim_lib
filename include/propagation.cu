//
// Created by jsmith on 10/2/15.
//

#include "propagation.h"
//#include <driver_types.h>
//#include <cuda_runtime_api.h>
/*
 *  Constructor
 */
Propagation::Propagation(int _x, int _y, std::string fuel_model_name, std::string fuel_moisture_name){
   // Point at appropriate rothdata file
   printf("Propagation Constructor\n");
   simulation_ = new FireSim(_x, _y, fuel_model_name, fuel_moisture_name);
   // Initialize neighbor data
   sim_size_ = _x * _y;
   sim_rows_ = _y;
   sim_cols_ = _x;
//   rothdata_ = (float*)malloc(simulation_->sim_dim_x_ *simulation_->sim_dim_y_ *3*sizeof(float));
   maxspreadrate_ = (float*) malloc(sim_size_*16*sizeof(float));
   curspreadrate_ = (float*) malloc(sim_size_*16*sizeof(float));
   intensity_modifier_ = (float*) malloc(sim_size_*sizeof(float));
   acceleration_constant_ = (float*) malloc(sim_size_*sizeof(float));
}

/*
 *  Destructor
 */
Propagation::~Propagation(){
//   rothdata_ = NULL;
   free(maxspreadrate_);
   free(curspreadrate_);
   free(intensity_modifier_);
   free(acceleration_constant_);
//   free(l_n_);
   // Free Cuda memory
//   cudaFree(g_rothdata_);
   cudaFree(g_maxspreadrate_);
   cudaFree(g_curspreadrate_);
   cudaFree(g_intensity_modifier_);
   cudaFree(g_acceleration_constat_);
   cudaFree(g_I_o_);
   cudaFree(g_RAC_);
   cudaFree(g_l_n_);

   delete simulation_;
}

/*
 *  Init()
 *  Functionality: Initializes all the values needed to be passed to the GPU.
 *                 This step is necessary for updating the simulation during the
 *                 runtime execution of the simulation.
 */
bool Propagation::Init(std::string fuel_file, std::string terrain_file,
                       std::string canopy_height_file, std::string crown_base_height_file,
                       std::string crown_bulk_density_file, float wind_x, float wind_y){
   printf("Propagation Init\n");
   max_spread = 0;
   /*
      cells formatting will be as follows:
      cell[x*8] is reference cell, following 8/16 vals are direction data
      N  NE   E  SE   S  SW   W  NW  NNW NNE NEE SEE SSE SSW SWW NWW
   */
   // Initialize Simulation
   simulation_->Init(fuel_file, terrain_file,
                     canopy_height_file, crown_base_height_file,
                     crown_bulk_density_file, wind_x, wind_y);
   simulation_->UpdateSpreadData();
   l_n_ = simulation_->l_n_;
   int fuel_model;

   float angles[16] = {  90,  45,  0, -45, -90, -115,  180,  115, 63.4349, 26.5651, -26.5651f,
                         -63.4349f, -116.5651f, -153.4349f, 153.4349f, 116.5651f};
   int dirs = 16;
   for(int row = 0, cell = 0; row < sim_rows_; row++) {
      for (int col = 0; col < sim_cols_; col++) {
         for (int i = 0; i < dirs; i++, cell++) {
            if(simulation_->roth_data_[row][col].x < 0) {
               maxspreadrate_[cell] = 0.0;
            }
            maxspreadrate_[cell] = (float) (simulation_->roth_data_[row][col].x * (1.0f - simulation_->roth_data_[row][col].z) /
                                               (1.0f - simulation_->roth_data_[row][col].z *
                                                       cos(simulation_->roth_data_[row][col].y * 3.14159f / 180.f - angles[i])));
            curspreadrate_[cell] = 0.f;

            if(maxspreadrate_[cell] > max_spread){
               max_spread = maxspreadrate_[cell];
               std::cout << max_spread << std::endl;
            }
         }
         fuel_model = simulation_->fuel_t_[cell/dirs];
//         std::cout << simulation_->fuel_t_[cell/dirs] << ' ' << fuel_model << ' ' << simulation_->fuel_sav_accel_b_[fuel_model].y << std::endl;
         intensity_modifier_[cell/dirs] = simulation_->roth_data_[row][col].w;
         acceleration_constant_[cell/dirs] = simulation_->fuel_sav_accel_b_[fuel_model].y;
      }
   }

   return true;
}

/*
 *  CopyToDevice()
 *  Functionality: Copies memory from host to device for propagation
 */
bool Propagation::CopyToDevice() {
   printf("Propgation Copy To Device\n");
//   cudaError_t err = cudaMalloc( (void**) &g_rothdata_, sim_size_*3*sizeof(float));
   cudaError_t err = cudaMalloc((void**) &g_maxspreadrate_, sim_size_ * 16 * sizeof(float));
   err = cudaMalloc((void**) &g_curspreadrate_, sim_size_ * 16 * sizeof(float));
   err = cudaMalloc((void**) &g_intensity_modifier_, sim_size_ * sizeof(float));
   err = cudaMalloc((void**) &g_acceleration_constat_, sim_size_ * sizeof(float));
   err = cudaMalloc((void**) &g_I_o_, sim_size_ * sizeof(float));
   err = cudaMalloc((void**) &g_RAC_, sim_size_ * sizeof(float));
   err = cudaMalloc((void**) &g_l_n_, 16 * sizeof(float));
   if (err != cudaSuccess) {
      std::cerr << "Error Allocating Memory in Propagation Class: " << cudaGetErrorString(err) << std::endl;
      exit(1);
   }
//   err = cudaMemcpy(g_rothdata_, rothdata_, sim_size_*3*sizeof(float), cudaMemcpyHostToDevice);
   err = cudaMemcpy(g_maxspreadrate_, maxspreadrate_, sim_size_*16*sizeof(float), cudaMemcpyHostToDevice);
   err = cudaMemcpy(g_curspreadrate_, curspreadrate_, sim_size_*16*sizeof(float), cudaMemcpyHostToDevice);
   err = cudaMemcpy(g_intensity_modifier_, intensity_modifier_, sim_size_*sizeof(float), cudaMemcpyHostToDevice);
   err = cudaMemcpy(g_I_o_, simulation_->I_o_, sim_size_*sizeof(float), cudaMemcpyHostToDevice);
   err = cudaMemcpy(g_RAC_, simulation_->RAC_, sim_size_*sizeof(float), cudaMemcpyHostToDevice);
   err = cudaMemcpy(g_l_n_, l_n_, 16*sizeof(float), cudaMemcpyHostToDevice);
   if (err != cudaSuccess) {
      std::cerr << "Error Copying in Propagation Class: " << cudaGetErrorString(err) << std::endl;
      exit(1);
   }
   return true;
}
