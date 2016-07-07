//
// Created by jsmith on 10/2/15.
//

#ifndef SIMULATOR_BD_H
#define SIMULATOR_BD_H

#include "propagation.h"
#include "kernel_BD.h"

class BD : Propagation
{
public:
   BD(int,int, std::string, std::string);
   ~BD();
   bool Init(std::string,std::string,std::string,std::string,std::string,float,float);
   bool CopyToDevice();
   bool RunKernel(int,int,int, bool);
   bool CopyFromDevice();
   bool WriteToFile(std::string);
   bool UpdateCell(int,int,int);

protected:
   // Host Variables
   float* toa_map_;
   float* timesteppers_;
   float* loc_burndist_;
   float timestep_;
   float current_time_;
   // Device Variables
   float* g_toa_map_in_;
   float* g_toa_map_out_;
   float* g_timesteppers_;
   float* g_loc_burndist_;
};
#endif //SIMULATOR_BD_H
