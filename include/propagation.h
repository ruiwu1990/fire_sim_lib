//
// Created by jsmith on 10/2/15.
//

#ifndef SIMULATOR_PROPAGATION_H
#define SIMULATOR_PROPAGATION_H
#include <iostream>
#include <fstream>
#include "FireSim.h"

class Propagation{
public:
    Propagation(int,int, std::string, std::string);
    ~Propagation();
    virtual bool Init(std::string,std::string,std::string,std::string,std::string,std::string,std::string);
    virtual bool CopyToDevice();
//      virtual bool Accelerate();
//      virtual bool RunKernel();
//      virtual bool CopyFromDevice();

protected:
    // Host Variables
    FireSim* simulation_;
//      float* rothdata_;
    float* maxspreadrate_;
    float* curspreadrate_;
    float* intensity_modifier_;
    float* acceleration_constant_;
    float* l_n_;
    int sim_size_;
    int sim_rows_;
    int sim_cols_;
    float max_spread;
    // Device Variables
//      float* g_rothdata_;
    float* g_maxspreadrate_;
    float* g_curspreadrate_;
    float* g_intensity_modifier_;
    float* g_acceleration_constat_;
    float* g_I_o_;
    float* g_RAC_;
    float* g_l_n_;
};
#endif //SIMULATOR_PROPAGATION_H