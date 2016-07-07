//
// Created by jsmith on 10/2/15.
//
//#include <driver_types.h>
#include "BD.h"

BD::BD(int _x, int _y, std::string fuel_model_name, std::string fuel_moisture_name)
        : Propagation(_x, _y, fuel_model_name, fuel_moisture_name){
//   printf("BD Constructor\n");
//   sim_size_ = simulation_->sim_dim_x_ * simulation_->sim_dim_y_;
   toa_map_ = (float*) malloc(sim_size_ * sizeof(float));
   timesteppers_ = (float*) malloc(2*sizeof(float));
   loc_burndist_ = (float*) malloc(8*sim_size_*sizeof(float));
}

BD::~BD(){
   // Free host memory
   free(toa_map_);
   free(timesteppers_);
   // Free device memory
   cudaFree(g_toa_map_in_);
   cudaFree(g_toa_map_out_);
   cudaFree(g_timesteppers_);
}

bool BD::Init(std::string fuel_file, std::string terrain_file,
              std::string canopy_height_file, std::string crown_base_height_file,
              std::string crown_bulk_density_file, float wind_x, float wind_y){
   Propagation::Init(fuel_file, terrain_file,
                     canopy_height_file, crown_base_height_file,
                     crown_bulk_density_file, wind_x, wind_y);
//   timestep_ = simulation_->time_step_;
   timestep_ = max_spread / simulation_->cell_size_;  // 1 timestep = 100 seconds
   std::cout << timestep_ << std::endl;
   printf("BD Init\n");
   for(unsigned int i = 0; i < sim_size_; i++){
      toa_map_[i] = simulation_->ign_time_[i];
   }
   // Populate Burn Distances
   for(unsigned int i = 0; i < sim_size_; i++){
      for(unsigned int j = 0; j < 8; j++){
         loc_burndist_[i*8+j] = simulation_->l_n_[j];
      }
   }
   timesteppers_[0] = 0;
   timesteppers_[1] = 0;
   current_time_ = 0.0f;
   return true;
}

bool BD::CopyToDevice(){
   Propagation::CopyToDevice();
//   for(int cell = 0; cell < sim_size_; cell++){
//      if(cell == 0)
//         printf("------------------------- FIIIIIREEEEEE ------------------\n\n");
//   }
//   printf("BD Copy To Device\n");
   // Create memory on device
   cudaError_t err = cudaMalloc( (void**) &g_toa_map_in_, sim_size_*sizeof(float));
   err = cudaMalloc( (void**) &g_toa_map_out_, sim_size_*sizeof(float));
   err = cudaMalloc( (void**) &g_timesteppers_, 2*sizeof(float));
   err = cudaMalloc( (void**) &g_loc_burndist_, sim_size_*8*sizeof(float));
   if (err != cudaSuccess) {
      std::cerr << "Error Allocating Memory in BD Class: " << cudaGetErrorString(err) << std::endl;
      exit(1);
      return false;
   }

   err = cudaMemcpy(g_toa_map_in_, toa_map_, sim_size_*sizeof(float), cudaMemcpyHostToDevice);
   err = cudaMemcpy(g_toa_map_out_, toa_map_, sim_size_*sizeof(float), cudaMemcpyHostToDevice);
   err = cudaMemcpy(g_timesteppers_, timesteppers_, 2*sizeof(float), cudaMemcpyHostToDevice);
   err = cudaMemcpy(g_loc_burndist_, loc_burndist_,sim_size_*8*sizeof(float), cudaMemcpyHostToDevice);
   if (err != cudaSuccess) {
      std::cerr << "Error Copying Memory in BD Class: " << cudaGetErrorString(err) << std::endl;
      exit(1);
      return false;
   }
   return true;
}

bool BD::RunKernel(int sim_step, int B, int T, bool crowning_test) {
   int counter = 0;
   while(counter < sim_step){
      counter++;
      // BURN DISTANCE METHOD
      // Do calculations
//          BurnDist<<<B,T>>>(g_toa_map_in_, g_toa_map_out_, g_loc_burndist_, g_maxspreadrate_,
       BurnDist<<<B,T>>>(g_toa_map_in_, g_toa_map_out_, g_loc_burndist_, g_maxspreadrate_,
                        g_curspreadrate_, g_timesteppers_, g_l_n_, sim_size_,
                        sim_rows_, sim_cols_, timestep_, current_time_);

      copyKernelBD<<<B,T>>>(g_toa_map_in_, g_toa_map_out_,
                            sim_size_);
      if(crowning_test)
         TestCrownRate<<<B,T>>>(g_curspreadrate_,g_maxspreadrate_,g_intensity_modifier_,sim_size_, g_I_o_, g_RAC_);
      Accelerate<<<B,T>>>(g_curspreadrate_, g_maxspreadrate_, acceleration_constant_, sim_size_ * 16, timestep_);

      cudaDeviceSynchronize();

      current_time_ += timestep_;
   }
   return true;
}


bool BD::CopyFromDevice() {
   printf("BD Copy From Device\n");
   cudaError_t err =  cudaMemcpy(toa_map_, g_toa_map_in_,
                                 sim_size_*sizeof(float),
                                 cudaMemcpyDeviceToHost);
   if (err != cudaSuccess) {
      std::cerr << "Error copying from GPU: " << cudaGetErrorString(err) << std::endl;
      exit(1);
      return false;
   }
   return true;
}

bool BD::WriteToFile(std::string filename, std::string* metaptr) {
   std::ofstream fout;
//   std::string filename;
//   filename += simulation_->root_path_;
//   filename += "out/BD_test.csv";
   fout.open(filename.c_str());

   // add metadata to the first eight lines
   for(int x = 0; x < 8; x++)
   {
      fout << metaptr[x];
   }
   fout << '\n';

   for(unsigned int i = 0; i < sim_size_; i++){
      if(i % simulation_->sim_dim_x_ == 0 && i !=0){
         fout << '\n';
      }
//      if(toa_map_[i] == INF){
//         fout << 0 << " ";
//      }
//      else
         fout << (int) toa_map_[i] << ",";
   }
   fout.close();
   return true;
}


bool BD::UpdateCell(int _x, int _y, int val){
//   if(_x < 0 || _y < 0 || _x > sim_rows_ || _y > sim_cols_)
//      return false;
   int cell = _x * sim_cols_ + _y;
//   std::cout << cell << std::endl;
   if(cell < 0 || cell > sim_size_)
      return false;
   toa_map_[cell] = val;
   return true;
}
