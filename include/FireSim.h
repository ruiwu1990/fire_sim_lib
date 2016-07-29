//
// Created by jsmith on 10/2/15.
//

#ifndef SIMULATOR_FIRESIM_H
#define SIMULATOR_FIRESIM_H
#include <iostream>
#include <vector>
#include "FuelModel.h"
#include "FuelMoisture.h"
#include <string>
// Include gdal libraries to parse .dem files
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include "vector.h"
#include <map>

//using namespace std;

class FireSim {
public:
   FireSim(int _x,int _y, std::string, std::string);
   ~FireSim();
   void     Init(std::string,std::string,std::string,std::string,std::string,std::string,std::string);
   void     UpdateSpreadData();
   float    Clamp(float, float, float);
   bool     BurnDistance(float &, float, float);
   bool     SetFire(int _x = 0, int _y = 0, int _time = 0);

   // Simulation Data
//    private:
   vec4**   roth_data_; // x - maxSpreadRate, y - spreadDirection, z - ellipseEccentricity, w - IntensityModifier
   float**  current_ros_;
   float**  toa_;
   int*   xwind_;
   int*   ywind_;
   float**  original_toa_;
   int**    update_stamp_;
   point**  source_data_texture_;

   float*   I_o_;
   float*   RAC_;
   float*   canopy_height_;
//   float**  spread_data_;
   float    acceleration_constant_;
   float    foliar_moisture;
   float**  intensity_modifier;

   // Simulation members from test Sim
   float    time_now_;
   float    time_next_;
   float*   ign_time_;
   float*   ign_time_new_;
   float**  burn_dist_;
   float*   l_n_;

   // Rothermel Data Members
   int*     fuel_t_;
   vec4*    dead_sav_burnable_b_;
   vec4*    dead_1h_b_;
   vec4*    dead_10h_b_;
   vec4*    dead_100h_b_;
   vec4*    live_h_b_;
   vec4*    live_w_b_;
   vec4*    fine_dead_extinctions_density_b_;
   vec4*    areas_reaction_factors_b_;
   vec4*    slope_wind_factors_b_;
   vec4*    residence_flux_live_sav_b_;
   vec2*    fuel_sav_accel_b_;
   vec3*    slope_aspect_elevation_t_;
   vec2**   wind_t_;
   vec3*    dead_moistures_t_;
   vec2*    live_moistures_t_;
   std::string   root_path_; // path pointing to correct directory
   char*    fuel_t_fname_;
   char*    slope_aspect_elevation_t_fname_;

   // Spotting Constants
   float C_d;
   float p_a;
   float p_s;
   float g;
   float K;
   float a_x;
   float b_x;
   float D_p; // 7mm diameter, just a random value I set
   int   B ;
   float v_o;
   float tau;

   int      sim_dim_x_;
   int      sim_dim_y_;
   float    cell_size_;
   float    time_step_;

   std::vector<sim::FuelModel> _models;
   std::vector<sim::FuelMoisture> _moistures;
};


#endif //SIMULATOR_FIRESIM_H
