#include <iostream>
#include <vector>
#include "FuelModel.h"
#include "FuelMoisture.h"
#include <string>
// Include gdal libraries to parse .dem files
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <gdal/cpl_conv.h>
#include "vector.h"

using namespace std;

class FireSim{
   public: 
      FireSim(int _x = 100,int _y = 100); // this will set to default test state
      ~FireSim();
      void Init();
      void UpdateSpreadData();
//      void propagateFire();
      void burnDistance();
      void accelerateFire();
      void triggerNextEvent();
      float Clamp(float, float, float);

      float*Accelerate(float *, float *, float);
      float TestCrownRate(float, float, float, float, float);
      void SetSimSize(int, int);
      bool BurnDistance(float &, float, float);

   // private: 

      // Simulation Data
      vec4**roth_data_; // x - maxSpreadRate, y - spreadDirection, z - ellipseEccentricity
      float**toa_;
      float**original_toa_;
      vec4**ortho_spread_rate_;
      vec4**diag_spread_rate_;
      vec4**ortho_max_spread_rate_;
      vec4**diag_max_spread_rate_;
      vec4**ortho_burn_dist_;
      vec4**diag_burn_dist_;
      int**update_stamp_;
      point**source_data_texture_;

      float**I_o_;
      float**RAC_;
      float**canopy_height_;
      float**spread_data_;

      float start_time_;
      float base_time_;
      float end_time_;
      int last_stamp_;
      int current_stamp_;
      float acceleration_constant_;
      
      float output_toa_;
      vec4*output_ortho_rates_;
      vec4*output_diag_rates_;
      int timestamp_;
      float*output_source_data_;

      // Simulation members from test Sim
      float time_now_;
      float time_next_;
      float*ign_time_;
      float*ign_time_new_;
      float**burn_dist_;
      float*l_n_;


      // Rothermel Data Members
      int*fuel_t_;
      vec4*dead_sav_burnable_b_;
      vec4*dead_1h_b_;
      vec4*dead_10h_b_;
      vec4*dead_100h_b_;
      vec4*live_h_b_;
      vec4*live_w_b_;
      vec4*fine_dead_extinctions_density_b_;
      vec4*areas_reaction_factors_b_;
      vec4*slope_wind_factors_b_;
      vec4*residence_flux_live_sav_b_;
      vec2*fuel_sav_accel_b_;
      // Roth textures
      vec3*slope_aspect_elevation_t_;
      vec2**wind_t_;
      vec3*dead_moistures_t_;
      vec2*live_moistures_t_;
      char*fuel_t_fname_;
      char*slope_aspect_elevation_t_fname_;

      int sim_dim_x_;
      int sim_dim_y_;
      float cell_size_;
      float time_step_;

      std::vector<sim::FuelModel> _models;
      std::vector<sim::FuelMoisture> _moistures;
};

