//
// Created by jsmith on 10/10/15.
//

#ifndef SIMULATOR_MT_H
#define SIMULATOR_MT_H
#include "propagation.h"
#include "kernel_MT.h"

class MT : Propagation{
public:
   MT(int,int, std::string, std::string);
   ~MT();
   bool Init(std::string,std::string,std::string,std::string,std::string,float,float);
   bool CopyToDevice();
   bool RunKernel(int, int, int, bool);
   bool CopyFromDevice();
   bool WriteToFile(std::string);
   bool UpdateCell(int,int,int);

protected:
   // Host Variabels
   int* toa_map_;
   int* timesteppers_;
   float* l_n_;
   // Device Variables
   int* g_toa_map_;
   int* g_timesteppers_;
};

#endif //SIMULATOR_MT_H
