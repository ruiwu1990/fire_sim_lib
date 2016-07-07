#include "fireSim_.h"
//#include "FireSim.h"
#include <algorithm>
#include <cmath>


const int INF = 9999999;

#define b printf("%u\n", __LINE__);


template<typename T> T* GISToFloatArray(char*, int, int);
template int* GISToFloatArray<int>(char*, int, int);
template float* GISToFloatArray<float>(char*, int, int);

/*
Constructor: builds simplest test case for testing code
*/
FireSim::FireSim(int _x, int _y){
   std::cout << "Initializing Simulation to Test Setting" << std::endl;

   // declare 2d map data
   sim_dim_x_ = _x;
   sim_dim_y_ = _y;

   _models = sim::readFuelModels("../data/default.fmd");
   _moistures = sim::readFuelMoistures("../data/kyle.fms");
   int numModels = _models.size();
   int numMoistModels = _moistures.size();

   toa_ = new float*[sim_dim_x_];
   roth_data_ = new vec4*[sim_dim_x_];
   original_toa_ = new float*[sim_dim_x_];
   ortho_spread_rate_ = new vec4*[sim_dim_x_];
   diag_spread_rate_ = new vec4*[sim_dim_x_];
   ortho_max_spread_rate_ = new vec4*[sim_dim_x_];
   diag_max_spread_rate_ = new vec4*[sim_dim_x_];
   ortho_burn_dist_ = new vec4*[sim_dim_x_];
   diag_burn_dist_ = new vec4*[sim_dim_x_];
   update_stamp_ = new int*[sim_dim_x_];
   source_data_texture_ = new point*[sim_dim_x_];

   I_o_ = new float*[sim_dim_x_];
   RAC_ = new float*[sim_dim_x_];
   canopy_height_ = new float*[sim_dim_x_];
   spread_data_ = new float*[sim_dim_x_];

   // rothermel vals
   fuel_t_ = NULL;
   slope_aspect_elevation_t_ = new vec3[sim_dim_x_ * sim_dim_y_];

   dead_sav_burnable_b_ = new vec4[numModels];
   dead_1h_b_ = new vec4[numModels];
   dead_10h_b_ = new vec4[numModels];
   dead_100h_b_ = new vec4[numModels];
   live_h_b_ = new vec4[numModels];
   live_w_b_ = new vec4[numModels];
   fine_dead_extinctions_density_b_ = new vec4[numModels];
   areas_reaction_factors_b_ = new vec4[numModels];
   slope_wind_factors_b_ = new vec4[numModels];
   residence_flux_live_sav_b_ = new vec4[numModels];
   fuel_sav_accel_b_ = new vec2[numModels];

   wind_t_ = new vec2*[sim_dim_x_];
   dead_moistures_t_ = new vec3[numMoistModels];
   live_moistures_t_ = new vec2[numMoistModels];


   for(int i = 0; i < sim_dim_x_; i++){
      toa_[i] = new float[sim_dim_y_];
      roth_data_[i] = new vec4[sim_dim_y_];
      original_toa_[i] = new float[sim_dim_y_];
      ortho_spread_rate_[i] = new vec4[sim_dim_y_];
      diag_spread_rate_[i] = new vec4[sim_dim_y_];
      ortho_max_spread_rate_[i] = new vec4[sim_dim_y_];
      diag_max_spread_rate_[i] = new vec4[sim_dim_y_];
      ortho_burn_dist_[i] = new vec4[sim_dim_y_];
      diag_burn_dist_[i] = new vec4[sim_dim_y_];
      update_stamp_[i] = new int[sim_dim_y_];
      source_data_texture_[i] = new point[sim_dim_y_];

      I_o_[i] = new float[sim_dim_y_];
      RAC_[i] = new float[sim_dim_y_];
      canopy_height_[i] = new float[sim_dim_y_];
      spread_data_[i] = new float[sim_dim_y_];

      // rothermel
      wind_t_[i] = new vec2[sim_dim_y_];
   }

   start_time_ = 0.f;
   base_time_ = 0.f;
   end_time_ = 1000.f;
   last_stamp_ = 0;
   current_stamp_ = 0;
   acceleration_constant_ = 1.0;

   fuel_t_fname_ = new char[18];
   fuel_t_fname_ = "../data/fixed.fuel";
   slope_aspect_elevation_t_fname_ = new char[17];
   slope_aspect_elevation_t_fname_ = "../data/fixed.dem";

   // Simulation Data Members
   time_now_ = 0.0;
   time_next_ = 0.0;
   ign_time_ = new float[sim_dim_x_ * sim_dim_y_];
   ign_time_new_ = new float[sim_dim_x_ * sim_dim_y_];
   burn_dist_ = new float*[sim_dim_x_ * sim_dim_y_];
   for(int i = 0; i < sim_dim_x_ * sim_dim_y_; i++){
      burn_dist_[i] = new float[8];
   }

   cell_size_ = 300;
   l_n_ = new float[16];
   float orthoSize = cell_size_;
   float diagSize = cell_size_ * sqrt(2);
   float superSize = sqrt(pow(cell_size_, 2) + pow(cell_size_ *2, 2));
   static float L_n_tmp[16] =  { orthoSize, diagSize, orthoSize, diagSize, orthoSize, diagSize,
                                 orthoSize, diagSize, superSize,superSize,superSize,superSize,
                                 superSize,superSize,superSize,superSize};
   for(int i = 0; i < 16; i++){
      l_n_[i] = L_n_tmp[i];
   }
   // cout << "constructed" << endl;
}

/*
Destructor: builds simplest test case for testing code
*/
FireSim::~FireSim(){
   // delete all memory: need to be more clever with more complex sims

   // blah blah
      // std::cout << "Deallocating memory" << std::endl;
   // simDimX = sizeX;
   // simDimY = sizeY;
   for(int i = 0; i < sim_dim_x_; i++){
      delete toa_[i];
      delete roth_data_[i];
      delete original_toa_[i];
      delete ortho_spread_rate_[i];
      delete diag_spread_rate_[i];
      delete ortho_max_spread_rate_[i];
      delete diag_max_spread_rate_[i];
      delete ortho_burn_dist_[i];
      delete diag_burn_dist_[i];
      delete update_stamp_[i];
      delete source_data_texture_[i];

      delete I_o_[i];
      delete RAC_[i];
      delete canopy_height_[i]; //// I am seg faulting... wtf.
      delete spread_data_[i];

      // rothermel
      delete wind_t_[i];
   }

   delete toa_;
   delete roth_data_;
   delete original_toa_;
   delete ortho_spread_rate_;
   delete diag_spread_rate_;
   delete ortho_max_spread_rate_;
   delete diag_max_spread_rate_;
   delete ortho_burn_dist_;
   delete diag_burn_dist_;
   delete update_stamp_;
   delete source_data_texture_;

   delete I_o_;
   delete RAC_;
   delete canopy_height_;
   delete spread_data_;

   // rothermel vals
   delete slope_aspect_elevation_t_;

   delete dead_sav_burnable_b_;
   delete dead_1h_b_;
   delete dead_10h_b_;
   delete dead_100h_b_;
   delete live_h_b_;
   delete live_w_b_;
   delete fine_dead_extinctions_density_b_;
   delete areas_reaction_factors_b_;
   delete slope_wind_factors_b_;
   delete residence_flux_live_sav_b_;
   delete fuel_sav_accel_b_;

   delete wind_t_;
   delete dead_moistures_t_;
   delete live_moistures_t_;



   start_time_ = 0.;
   base_time_ = 0.;
   end_time_ = 1000.;
   last_stamp_ = 0.;
   current_stamp_ = 0.;
   acceleration_constant_ = 1.0;

   // delete fuelTextureFile;
   // delete slopeAspectElevationTextureFile;

      // cout << "test" << endl;
   // Simulation Data Members
   time_now_ = 0.0;
   time_next_ = 0.0;
   delete ign_time_;
   delete ign_time_new_;
   for(int i = 0; i < sim_dim_x_ * sim_dim_y_; i++){
      delete burn_dist_[i];
   }
   delete burn_dist_;
}



/*
Function: Init
Input: TBD
Shader base: rothermel
Purpose: Initializes the sim. 
*/
void FireSim::Init(){
   // read from files:
   int cell = 0;
   float* slopeTexTmp = NULL;
   GDALAllRegister();
   fuel_t_ = GISToFloatArray<int>(fuel_t_fname_, sim_dim_x_, sim_dim_y_);
   slopeTexTmp = GISToFloatArray<float>(slope_aspect_elevation_t_fname_, sim_dim_x_ *3, sim_dim_y_ *3);

   for(int i = 0; i < sim_dim_x_; i++){
      for(int j = 0; j < sim_dim_y_; j++, cell++){
         toa_[i][j] = 20.;
         roth_data_[i][j].x = roth_data_[i][j].y = roth_data_[i][j].z = 0.;
         // fuelTexture[i][j] = 0.;
         original_toa_[i][j] = 20.;
         ortho_spread_rate_[i][j].x = ortho_spread_rate_[i][j].y = ortho_spread_rate_[i][j].z = ortho_spread_rate_[i][j].w = 1.;
         diag_spread_rate_[i][j].x = diag_spread_rate_[i][j].y = diag_spread_rate_[i][j].z = diag_spread_rate_[i][j].w = 1.;
         ortho_max_spread_rate_[i][j].x = ortho_max_spread_rate_[i][j].y = ortho_max_spread_rate_[i][j].z = ortho_max_spread_rate_[i][j].w = 100.;
         diag_max_spread_rate_[i][j].x = diag_max_spread_rate_[i][j].y = diag_max_spread_rate_[i][j].z = diag_max_spread_rate_[i][j].w = 100.;
         ortho_burn_dist_[i][j].x = ortho_burn_dist_[i][j].y = ortho_burn_dist_[i][j].z = ortho_burn_dist_[i][j].w = .2;
         diag_burn_dist_[i][j].x = diag_burn_dist_[i][j].y = diag_burn_dist_[i][j].z = diag_burn_dist_[i][j].w = .2;
         update_stamp_[i][j] = 0.;
         source_data_texture_[i][j].x = source_data_texture_[i][j].y = 0.;

         I_o_[i][j] = 0.0;
         RAC_[i][j] = 100000.;
         canopy_height_[i][j] = 0.;
         spread_data_[i][j] = 0.;

         // Rothermel Data Members
         wind_t_[i][j].x = wind_t_[i][j].y = 0.;

         slope_aspect_elevation_t_[cell].x = slopeTexTmp[3*cell];
         slope_aspect_elevation_t_[cell].y = slopeTexTmp[3*cell+1];
         slope_aspect_elevation_t_[cell].z = slopeTexTmp[3*cell+2];

         ign_time_[cell] = INF;
         ign_time_new_[cell] = INF;
         for(int k = 0; k < 8; k++){
            burn_dist_[cell][k] = l_n_[k];
         }
      }
   }

   spread_data_[5][5] = 100;
   int ignSpot = sim_dim_x_ * sim_dim_y_ / 2 + sim_dim_y_ / 2;
   ign_time_[ignSpot] = 0;
   ign_time_new_[ignSpot] = 0;
   // ignTime[ignSpot/2] = 0;
   // ignTimeNew[ignSpot/2] = 0;
   time_step_ = 2.0;


   int i = 0;
   for (std::vector<sim::FuelModel>::iterator it = _models.begin(); 
        it != _models.end(); it++, i++)
   {
      dead_1h_b_[i].x = it->effectiveHeatingNumber[sim::Dead1h];
      dead_1h_b_[i].y = it->load[sim::Dead1h];
      dead_1h_b_[i].z = it->areaWeightingFactor[sim::Dead1h];
      dead_1h_b_[i].w = it->fuelMoisture[sim::Dead1h];
      
      dead_10h_b_[i].x = it->effectiveHeatingNumber[sim::Dead10h];
      dead_10h_b_[i].y = it->load[sim::Dead10h];
      dead_10h_b_[i].z = it->areaWeightingFactor[sim::Dead10h];
      dead_10h_b_[i].w = it->fuelMoisture[sim::Dead10h];
      
      dead_100h_b_[i].x = it->effectiveHeatingNumber[sim::Dead100h];
      dead_100h_b_[i].y = it->load[sim::Dead100h];
      dead_100h_b_[i].z = it->areaWeightingFactor[sim::Dead100h];
      dead_100h_b_[i].w = it->fuelMoisture[sim::Dead100h];
      
      live_h_b_[i].x = it->effectiveHeatingNumber[sim::LiveH];
      live_h_b_[i].y = it->load[sim::LiveH];
      live_h_b_[i].z = it->areaWeightingFactor[sim::LiveH];
      live_h_b_[i].w = it->fuelMoisture[sim::LiveH];
      
      live_w_b_[i].x = it->effectiveHeatingNumber[sim::LiveW];
      live_w_b_[i].y = it->load[sim::LiveW];
      live_w_b_[i].z = it->areaWeightingFactor[sim::LiveW];
      live_w_b_[i].w = it->fuelMoisture[sim::LiveW];

      fine_dead_extinctions_density_b_[i].x = it->fineDeadRatio;
      fine_dead_extinctions_density_b_[i].y = it->extinctionMoisture;
      fine_dead_extinctions_density_b_[i].z = it->liveExtinction;
      fine_dead_extinctions_density_b_[i].w = it->fuelDensity;

      areas_reaction_factors_b_[i].x = it->deadArea;
      areas_reaction_factors_b_[i].y = it->liveArea;
      areas_reaction_factors_b_[i].z = it->deadReactionFactor;
      areas_reaction_factors_b_[i].w = it->liveReactionFactor;

      slope_wind_factors_b_[i].x = it->slopeK;
      slope_wind_factors_b_[i].y = it->windK;
      slope_wind_factors_b_[i].z = it->windB;
      slope_wind_factors_b_[i].w = it->windE;

      residence_flux_live_sav_b_[i].x = it->residenceTime;
      residence_flux_live_sav_b_[i].y = it->propagatingFlux;
      residence_flux_live_sav_b_[i].z = it->SAV[sim::LiveH];
      residence_flux_live_sav_b_[i].w = it->SAV[sim::LiveW];

      dead_sav_burnable_b_[i].x = it->SAV[sim::Dead1h];
      dead_sav_burnable_b_[i].y = it->SAV[sim::Dead10h];
      dead_sav_burnable_b_[i].z = it->SAV[sim::Dead100h];
      // deadSAVBurnableBuffer[i].w = it->burnable? 100.0f : 0.0f;
      dead_sav_burnable_b_[i].w = 100.0f;

      fuel_sav_accel_b_[i].x = it->fuelSAV;
      fuel_sav_accel_b_[i].y = it->accelerationConstant;
   }

   i = 0;
   for (std::vector<sim::FuelMoisture>::iterator it = _moistures.begin(); 
        it != _moistures.end(); it++, i++)
   {         
         dead_moistures_t_[i].x = it->dead1h;
         dead_moistures_t_[i].y = it->dead10h;
         dead_moistures_t_[i].z = it->dead100h;

         live_moistures_t_[i].x = it->liveH;
         live_moistures_t_[i].y = it->liveW;
   }
}

/*
Function: updateSpreadData
Input: The necessary inputs are the values that are found in the textures/buffers
       in the FuelModel.h/Simulator.cpp files in Roger's code
Shader base: rothermel
Purpose: This runs rothermel's equations to initialize simulation
*/
void FireSim::UpdateSpreadData(){
   // This is where rothermel's shader is implemented

   cout << "Updating Spread Data . . ." << endl;
   int cell = 0;   /* row, col, and index of neighbor cell */
   float dist = 10.;
   int counter = 0;
   
   for(int i = 0; i < sim_dim_x_; i++){
      for(int j = 0; j < sim_dim_y_; j++, cell++){
         // Shader code: int fuelModel = texture2D(fuelTexture, gl_TexCoord[1].st).r;
            // gl_TexCoord[1].st corresponds to fuelTexture.xy
         // int fuelModel = fuelTexture[cell];
         // cout << "FUEL MODEL " << fuelModel << endl;

         // FOR TESTING: Must fix interpolation of fuelModel data
         int fuelModel = 1;

         vec4 dead1h, deadSAVBurnable, dead10h, dead100h, liveH, 
              liveW, fineDeadExtinctionsDensity, areasReactionFactors,
              slopeWindFactors, residenceFluxLiveSAV;
         vec2 fuelSAVAccel;

         vec3 slopeAspectElevation;
         vec2 wind;
         vec3 deadMoistures;
         vec2 liveMoistures;

         // Get data into vars
         // cout << deadSAVBurnableBuffer[fuelModel].x << " " << 
         //         deadSAVBurnableBuffer[fuelModel].y << " " <<
         //         deadSAVBurnableBuffer[fuelModel].z << " " <<
         //         deadSAVBurnableBuffer[fuelModel].w << endl;
         deadSAVBurnable = dead_sav_burnable_b_[fuelModel];
         // cout << deadSAVBurnable.x << " " << 
         //         deadSAVBurnable.y << " " <<
         //         deadSAVBurnable.z << " " <<
         //         deadSAVBurnable.w << endl;
         if(deadSAVBurnable.w < 50.0){
            cout << "skipping" << endl;
            continue;
         }


         dead1h = dead_1h_b_[fuelModel];
         dead10h = dead_10h_b_[fuelModel];
         dead100h = dead_100h_b_[fuelModel];
         liveH = live_h_b_[fuelModel];
         liveW = live_w_b_[fuelModel];
         fineDeadExtinctionsDensity = fine_dead_extinctions_density_b_[fuelModel];
         areasReactionFactors = areas_reaction_factors_b_[fuelModel];
         slopeWindFactors = slope_wind_factors_b_[fuelModel];
         residenceFluxLiveSAV = residence_flux_live_sav_b_[fuelModel];
         fuelSAVAccel = fuel_sav_accel_b_[fuelModel];
         float fuelSAV = fuelSAVAccel.x;
         float accelerationConstant = fuelSAVAccel.y;

         slopeAspectElevation = slope_aspect_elevation_t_[cell];

         wind = wind_t_[i][j];
         deadMoistures = dead_moistures_t_[fuelModel];
         liveMoistures = live_moistures_t_[fuelModel];

         float maxSpreadRate = 0.;
         float ellipseEccentricity = 0.;
         float spreadDirection = 0.;
         float spreadModifier = 0.;
         vec3 timeLagClass;


         if (deadSAVBurnable.x > 192.0)
            timeLagClass.x = deadMoistures.x;
         else if (deadSAVBurnable.x > 48.0)
            timeLagClass.x = deadMoistures.y;
         else
            timeLagClass.x = deadMoistures.z;

         if (deadSAVBurnable.y > 192.0)
            timeLagClass.y = deadMoistures.x;
         else if (deadSAVBurnable.y > 48.0)
            timeLagClass.y = deadMoistures.y;
         else
            timeLagClass.y = deadMoistures.z;

         if (deadSAVBurnable.z > 192.0)
            timeLagClass.z = deadMoistures.x;
         else if (deadSAVBurnable.z > 48.0)
            timeLagClass.z = deadMoistures.y;
         else
            timeLagClass.z = deadMoistures.z;

         float weightedFuelModel = 
            timeLagClass.x * dead1h.x * dead1h.y +
            timeLagClass.y * dead10h.x * dead10h.y +
            timeLagClass.z * dead100h.x * dead100h.y;

         float fuelMoistures[5];
         fuelMoistures[0] = timeLagClass.x;
         fuelMoistures[1] = timeLagClass.y;
         fuelMoistures[2] = timeLagClass.z;
         fuelMoistures[3] = liveMoistures.x;
         fuelMoistures[4] = liveMoistures.y;
         // for(int c = 0; c < 5; c++){
         //    cout << fuelMoistures[c] << endl;
         // }

         float liveExtinction = 0.0;
         if(liveH.y > 0.0 || liveW.y > 0.0){
            float fineDeadMoisture = 0.0;
            if (fineDeadExtinctionsDensity.x > 0.0)
               fineDeadMoisture = weightedFuelModel / fineDeadExtinctionsDensity.x;

            liveExtinction =
               (fineDeadExtinctionsDensity.z * 
                (1.0 - fineDeadMoisture / fineDeadExtinctionsDensity.y)) - 0.226;
            liveExtinction = max(liveExtinction, fineDeadExtinctionsDensity.y);
         }
         
         float heatOfIgnition =
            areasReactionFactors.x *
               ((250.0 + 1116.0 * fuelMoistures[0]) * dead1h.z * dead1h.x +
                (250.0 + 1116.0 * fuelMoistures[1]) * dead10h.z * dead10h.x +
                (250.0 + 1116.0 * fuelMoistures[2]) * dead100h.z * dead100h.x) +
            areasReactionFactors.y *
               ((250.0 + 1116.0 * fuelMoistures[3]) * liveH.z * liveH.x +
                (250.0 + 1116.0 * fuelMoistures[4]) * liveW.z * liveW.x);
         heatOfIgnition *= fineDeadExtinctionsDensity.w;

         float liveMoisture = liveH.z * fuelMoistures[3] + liveW.z * fuelMoistures[4];
         float deadMoisture = dead1h.z * fuelMoistures[0] + 
                              dead10h.z * fuelMoistures[1] + 
                              dead100h.z * fuelMoistures[2];

         float reactionIntensity = 0.0;

         if (liveExtinction > 0.0)
            {
               float r = liveMoisture / liveExtinction;
               if (r < 1.0)
                  reactionIntensity += areasReactionFactors.w * (1.0 - 
                                                                 (2.59 * r) + 
                                                                 (5.11 * r * r) - 
                                                      (3.52 * r * r * r));
            }
            if (fineDeadExtinctionsDensity.y > 0.0)
            {
               float r = deadMoisture / fineDeadExtinctionsDensity.y;
               if (r < 1.0)
                  reactionIntensity += areasReactionFactors.z * (1.0 - 
                                                                 (2.59 * r) + 
                                                                 (5.11 * r * r) - 
                                                      (3.52 * r * r * r));
            }

            float heatPerUnitArea = reactionIntensity * residenceFluxLiveSAV.x;
            float baseSpreadRate = 0.0;

            if (heatOfIgnition > 0.0)
               baseSpreadRate = reactionIntensity * residenceFluxLiveSAV.y / heatOfIgnition;
            // cout << "baseSpreadRate" << baseSpreadRate << endl;
            
            float slopeFactor = slopeWindFactors.x * slopeAspectElevation.x * slopeAspectElevation.x;
            float windFactor = 0.0;
            if (wind.x > 0.0)
               windFactor = slopeWindFactors.y * pow(wind.x, slopeWindFactors.z);

            spreadModifier = slopeFactor + windFactor;
            // cout << slopeFactor << " " << windFactor << endl;
            float upslope;
            if (slopeAspectElevation.y >= 180.0)
               upslope = slopeAspectElevation.y - 180.0;
            else
               upslope = slopeAspectElevation.y + 180.0;

            int checkEffectiveWindspeed = 0;
            int updateEffectiveWindspeed = 0;
            float effectiveWindspeed = 0.0;
            if (baseSpreadRate <= 0.0)
            {
               maxSpreadRate = 0.0;
               spreadDirection = 0.0;
// b
            }
            else if (spreadModifier <= 0.0)
            {
               maxSpreadRate = baseSpreadRate;
               spreadDirection = 0.0;
// b
            }
            else if (slopeAspectElevation.x <= 0.0)
            {
               effectiveWindspeed = wind.x;
               maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
               spreadDirection = wind.y;
               checkEffectiveWindspeed = 1;
// b
            }
            else if (wind.x <= 0.0)
            {
               maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
               spreadDirection = upslope;
               updateEffectiveWindspeed = 1;
               checkEffectiveWindspeed = 1;
// b
            }
            else if (fabs(wind.y - upslope) < 0.000001)
            {
               maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
               spreadDirection = upslope;
               updateEffectiveWindspeed = 1;
               checkEffectiveWindspeed = 1;
// b
            }
            else
            {
               float angleDelta;
               if (upslope <= wind.y)
                  angleDelta = wind.y - upslope;
               else
                  angleDelta = 360.0 - upslope + wind.y;
               angleDelta *= 3.14159 / 180.0;
               float slopeRate = baseSpreadRate * slopeFactor;
               float windRate = baseSpreadRate * windFactor;
               float x = slopeRate + windRate * cos(angleDelta);
               float y = windRate * sin(angleDelta);
               float addedSpeed = sqrt(x * x + y * y);
               maxSpreadRate = baseSpreadRate + addedSpeed;

               spreadModifier = maxSpreadRate / baseSpreadRate - 1.0;
               // cout << "spreadmoid: " << spreadModifier << endl;
               if (spreadModifier > 0.0)
                  updateEffectiveWindspeed = 1;
               checkEffectiveWindspeed = 1;

               float addedAngle = 0.0;
               if (addedSpeed > 0.0)
                  addedAngle = asin(clamp(fabs(y) / addedSpeed, -1.0, 1.0));
               float angleOffset = 0.0;
               if (x >= 0.0)
               {
                  if (y >= 0.0)
                     angleOffset = addedAngle;
                  else
                     angleOffset = 2.0 * 3.14159 - addedAngle;
               }
               else
               {
                  if (y >= 0.0)
                     angleOffset = 3.14159 + addedAngle;
                  else
                     angleOffset = 3.14159 - angleOffset;
               }
               spreadDirection = upslope + angleOffset * 180.0 / 3.14159;
               if (spreadDirection > 360.0)
                  spreadDirection -= 360.0;
            }

            if (updateEffectiveWindspeed == 1)
            {
               effectiveWindspeed = pow((spreadModifier * slopeWindFactors.w), (1.0 / slopeWindFactors.z));
            }
            if (checkEffectiveWindspeed == 1)
            {
               float maxWind = 0.9 * reactionIntensity;
               if (effectiveWindspeed > maxWind)
               {
                  if (maxWind <= 0.0)
                     spreadModifier = 0.0;
                  else
                     spreadModifier = slopeWindFactors.y * pow(maxWind, slopeWindFactors.z);
                  maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
                  effectiveWindspeed = maxWind;
               }
            }
            ellipseEccentricity = 0.0;
            if (effectiveWindspeed > 0.0)
            {
               float lengthWidthRatio = 1.0 + 0.002840909 * effectiveWindspeed;
               ellipseEccentricity = sqrt(lengthWidthRatio * lengthWidthRatio - 1.0) / lengthWidthRatio;
            }
            //maxSpreadRate = maxSpreadRate * (1.0 - exp(-accelerationConstant * burnTime / 60.0));
            //float firelineIntensity = 
            // 3.4613 * (384.0 * (reactionIntensity / 0.189275)) * 
            //    (maxSpreadRate * 0.30480060960) / (60.0 * fuelSAV);
            //firelineIntensity =
            // reactionIntensity * 12.6 * maxSpreadRate / (60.0 * fuelSAV);
            float intensityModifier =
               3.4613 * (384.0 * (reactionIntensity / 0.189275)) *
                  (0.30480060960) / (60.0 * fuelSAV);
            // gl_FragData[0] = vec4(maxSpreadRate, 
            //                       ellipseEccentricity, spreadDirection, intensityModifier);

            roth_data_[i][j].x = maxSpreadRate;
            // cout << maxSpreadRate;
            roth_data_[i][j].y = spreadDirection;
            // cout << spreadDirection;
            roth_data_[i][j].z = ellipseEccentricity;
            // cout << ellipseEccentricity << endl;

      }
   }
   // cout << counter << endl;
}

/*
Function: propagateFire
Input: TBD
Shader base: propagateAccel
Purpose: 

void FireSim::propagateFire(){
   // must loop through all points in lattice
   for(int i = 0; i < simDimX; i++){ // loop through rows
      for(int j = 0; j < simDimY; j++){ // loop through cols
         point n,s,e,w,nw,sw,ne,se;
         bool toprow = false;
         bool bottomrow = false;
         bool leftcol = false;
         bool rightcol = false;
         
         float toa = timeOfArrival[i][j];
         // cout << "before cont" << endl;
         if(toa <= startTime)
            continue;
         // cout << "after cont" << endl;
         float sourceData[4] = {sourceDataTexture[i][j].x, sourceDataTexture[i][j].y, i, j};
         point sourceDir;
         sourceDir.x = sourceData[0];
         sourceDir.y = sourceData[1];
         float originalToa = originalTimeOfArrival[i][j];
         float* orthoBurnDistances;
         orthoBurnDistances = new float[4];
         orthoBurnDistances[0] = orthoBurnDistance[i][j].x;
         orthoBurnDistances[1] = orthoBurnDistance[i][j].y;
         orthoBurnDistances[2] = orthoBurnDistance[i][j].z;
         orthoBurnDistances[3] = orthoBurnDistance[i][j].w;
         float* diagBurnDistances;
         diagBurnDistances = new float[4];
         diagBurnDistances[0] = diagBurnDistance[i][j].x;
         diagBurnDistances[1] = diagBurnDistance[i][j].y;
         diagBurnDistances[2] = diagBurnDistance[i][j].z;
         diagBurnDistances[3] = diagBurnDistance[i][j].w;

         // check x bounds
         if(i-1 >= 0){
            nw.x = i-1;
            n.x = i-1;
            ne.x = i-1;
            toprow = true;
         }
         else{
            nw.x = 0;
            n.x = 0;
            ne.x = 0;
            toprow = false;
         }
         if(i+1 < simDimX){
            sw.x = i+1;
            s.x = i+1;
            se.x = i+1;
            bottomrow = true;
         }
         else{
            sw.x = 0;
            s.x = 0;
            se.x = 0;
            bottomrow = false;
         }
         w.x = i;
         e.x = i;
         // check y bounds
         if(j-1 >=0){
            nw.y = j-1;
            w.y = j-1;
            sw.y = j-1;
            leftcol = true;
         }
         else{
            nw.y = 0;
            w.y = 0;
            sw.y = 0;
            leftcol = false;
         }
         if(j+1 < simDimY){
            ne.y = j+1;
            e.y = j+1;
            se.y = j+1;
            rightcol = true;
         }
         else{
            ne.y = 0;
            e.y = 0;
            se.y = 0;
            rightcol = false;

         }
         n.y = j;
         s.y = j;
         // if(toprow == true){
         //    bool updatenw = updateStamp[nw.x][nw.y] == lastStamp;
         //    bool updaten = updateStamp[n.x][n.y] == lastStamp;
         //    bool updatene = updateStamp[ne.x][ne.y] == lastStamp;
         // }
         // if(bottomrow == true){
         //    bool updatesw = updateStamp[sw.x][sw.y] == lastStamp;
         //    bool updates = updateStamp[s.x][s.y] == lastStamp;
         //    bool updatese = updateStamp[se.x][se.y] == lastStamp;
         // }
         // if(leftcol == true)
         //    bool updatew = updateStamp[s.x][s.y] == lastStamp;
         // if(rightcol == true)
         //    bool updatese = updateStamp[se.x][se.y] == lastStamp;

         // check if any updates necessary
         // if(!(updatenw | updaten | updatene | updatew | updatee | updatesw | updates | updatese))
         //    continue;

         bool toaCorrupt = updateStamp[sourceDir.x][sourceDir.y] == lastStamp;
         float newToa = toa;
         float toaLimit = toa;
         if(toaCorrupt || lastStamp == 0){
            newToa = originalToa;
            toaLimit = originalToa;
         }
         // REMEMBER THAT X and Y correspond to pointCoord!!
         point direction;
         direction.x = i;
         direction.y = j;
         float newRate = 0.0;
         float dt = 0.0;

         // check for boundaries as you propagate
         // if(toprow == true){
            // update NW
            {
               float t = timeOfArrival[nw.x][nw.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = diagSpreadRate[nw.x][nw.y].w;
                  float dist = diagBurnDistances[0];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = nw.x;
                     direction.y = nw.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update N
            {
               float t = timeOfArrival[n.x][n.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = orthoSpreadRate[n.x][n.y].w;
                  float dist = orthoBurnDistances[0];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = n.x;
                     direction.y = n.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update NE
            {
               float t = timeOfArrival[ne.x][ne.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = diagSpreadRate[ne.x][ne.y].z;
                  float dist = diagBurnDistances[1];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = ne.x;
                     direction.y = ne.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
         // }
         // if()
            // Update W
            {
               float t = timeOfArrival[w.x][w.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = orthoSpreadRate[w.x][w.y].z;
                  float dist = orthoBurnDistances[1];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = w.x;
                     direction.y = w.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update E
            {
               float t = timeOfArrival[e.x][e.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = orthoSpreadRate[e.x][e.y].y;
                  float dist = orthoBurnDistances[2];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = e.x;
                     direction.y = e.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update SW
            {
               float t = timeOfArrival[sw.x][sw.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = diagSpreadRate[sw.x][sw.y].y;
                  float dist = diagBurnDistances[2];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = sw.x;
                     direction.y = sw.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update S
            {
               float t = timeOfArrival[s.x][s.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = orthoSpreadRate[s.x][s.y].x;
                  float dist = orthoBurnDistances[3];
                  float burnTime = dist / rate;
                  t += burnTime;
                  if(t < newToa){
                     newToa = t;
                     direction.x = s.x;
                     direction.y = s.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }
            // Update SE
               // cout << "OUT" << endl;
            {
               float t = timeOfArrival[se.x][se.y];
               if(t < endTime){
                  t = max(baseTime, t);
                  float rate = diagSpreadRate[se.x][se.y].x;
                  float dist = diagBurnDistances[3];
                  float burnTime = dist / rate;
                  t += burnTime;
                  // cout << t << endl;
                  if(t < newToa){
                     newToa = t;
                     direction.x = se.x;
                     direction.y = se.y;
                     dt = burnTime;
                     newRate = rate;
                  }
               }
            }

            // if(newToa >= toaLimit || newToa > endTime)
            //    continue;
            // cout << "test" << endl;
            float maxOrthoRates[4] = {orthoMaxSpreadRate[i][j].x,
                                      orthoMaxSpreadRate[i][j].y,
                                      orthoMaxSpreadRate[i][j].z,
                                      orthoMaxSpreadRate[i][j].w};
            float maxDiagRates[4] = {diagMaxSpreadRate[i][j].x,
                                     diagMaxSpreadRate[i][j].y,
                                     diagMaxSpreadRate[i][j].z,
                                     diagMaxSpreadRate[i][j].w};
            float* currentOrthoRates;
            currentOrthoRates = new float[4];
            currentOrthoRates[0] = diagSpreadRate[direction.x][direction.y].x;
            currentOrthoRates[1] = diagSpreadRate[direction.x][direction.y].y;
            currentOrthoRates[2] = diagSpreadRate[direction.x][direction.y].z;
            currentOrthoRates[3] = diagSpreadRate[direction.x][direction.y].w;

            float* currentDiagRates;
            currentDiagRates = new float[4];
            currentDiagRates[0] = diagSpreadRate[direction.x][direction.y].x;
            currentDiagRates[1] = diagSpreadRate[direction.x][direction.y].y;
            currentDiagRates[2] = diagSpreadRate[direction.x][direction.y].z;
            currentDiagRates[3] = diagSpreadRate[direction.x][direction.y].w;

            float _canopyHeight = canopyHeight[i][j];
            if(_canopyHeight > 0.0){
               float _crownActiveRate = crownActiveRate[i][j];
               float _crownThreshold = crownThreshold[i][j];
               float intensityModifier = spreadData[i][j];
               // Ortho Rates
               maxOrthoRates[0] = testCrownRate(currentOrthoRates[0],
                                                maxOrthoRates[0],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxOrthoRates[1] = testCrownRate(currentOrthoRates[1],
                                                maxOrthoRates[1],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxOrthoRates[2] = testCrownRate(currentOrthoRates[2],
                                                maxOrthoRates[2],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxOrthoRates[3] = testCrownRate(currentOrthoRates[3],
                                                maxOrthoRates[3],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               // Diag Rates
               maxDiagRates[0] = testCrownRate(currentDiagRates[0],
                                                maxDiagRates[0],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxDiagRates[1] = testCrownRate(currentDiagRates[1],
                                                maxDiagRates[1],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxDiagRates[2] = testCrownRate(currentDiagRates[2],
                                                maxDiagRates[2],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
               maxDiagRates[3] = testCrownRate(currentDiagRates[3],
                                                maxDiagRates[3],
                                                intensityModifier,
                                                _crownActiveRate,
                                                _crownThreshold);
            }

            currentOrthoRates = accelerate(currentOrthoRates, maxOrthoRates, dt);
            currentDiagRates = accelerate(currentDiagRates, maxDiagRates, dt);

            // Write results
            timeOfArrival[i][j] = newToa;
            diagSpreadRate[i][j].x = currentDiagRates[0];
            diagSpreadRate[i][j].y = currentDiagRates[1];
            diagSpreadRate[i][j].z = currentDiagRates[2];
            diagSpreadRate[i][j].w = currentDiagRates[3];
            orthoSpreadRate[i][j].x = currentOrthoRates[0];
            orthoSpreadRate[i][j].y = currentOrthoRates[1];
            orthoSpreadRate[i][j].z = currentOrthoRates[2];
            orthoSpreadRate[i][j].w = currentOrthoRates[3];

            timeStamp = currentStamp;
            sourceDataTexture[i][j].x = newRate;

      }
   }
}*/

/*
Function: burnDistance
Input: TBD
Shader base: partialBurn
Purpose: 
*/
void FireSim::burnDistance(){

}

/*
Function: accelerateFire
Input: TBD
Shader base: acceleration
Purpose: 
*/
void FireSim::accelerateFire()   {

}

/*
Function: triggerNextEvent
Input: TBD
Purpose: Step time through simulation
*/
void FireSim::triggerNextEvent(){

}


/*//////////////////////////////////////////////////////////////////
                     Support Functions
//////////////////////////////////////////////////////////////////*/


/*
Function: accelerate
Input: TBD
Purpose: run acceleration code
*/
float* FireSim::Accelerate(float *current, float *maxRate, float dt){
   for(int i = 0; i < 4; i++){
      current[i] = min(current[i], maxRate[i]);
   }
   // clamp
   float ratio[4];
   for(int i = 0; i < 4; i++){
      float tmp = current[i] / maxRate[i];
      if(tmp < 0.1){
         ratio[i] = 0.1;
      }
      else if(tmp > 0.9){
         ratio[i] = 0.9;
      }
      else
         ratio[i] = tmp;
   }

   // find timeToMax
   float timeToMax[4];
   for(int i = 0; i < 4; i++){
      timeToMax[i] = -log(1.0 - ratio[i]) / acceleration_constant_;
   }

   // clamp
   for(int i = 0; i < 4; i++){
      float tmp = dt / timeToMax[i];
      if(tmp < 0.0){
         tmp = 0.0;
      }
      else if(tmp > 1.0){
         tmp = 1.0;
      }

      current[i] = tmp * (maxRate[i] - current[i]) + current[i];
   }

   return current;
}

/*
Function: testCrownRate
Input: TBD
Purpose: tests crown rate in each update step
*/
float FireSim::TestCrownRate(float spreadRate,
                             float maxRate,
                             float intensityModifier,
                             float crownActiveRate,
                             float crownThreshold)
{
   if(maxRate <= 0.0)
      return 0.0;

   spreadRate *= 60.0;
   spreadRate /= 3.28;
   maxRate *= 60.0;
   maxRate /= 3.28;
   float firelineIntensity = spreadRate * intensityModifier;
   if(firelineIntensity > crownThreshold){
      float maxCrownRate = 3.34 * maxRate;
      float surfaceFuelConsumption = crownThreshold * spreadRate / firelineIntensity;
      float crownCoefficient = -log(0.1)/(0.9 * (crownActiveRate - surfaceFuelConsumption));
      float crownFractionBurned = 1.0 - exp(-crownCoefficient * (spreadRate - surfaceFuelConsumption));
      if(crownFractionBurned < 0.0)
         crownFractionBurned = 0.0;
      if(crownFractionBurned > 1.0)
         crownFractionBurned = 1.0;
      float crownRate = spreadRate + crownFractionBurned * (maxCrownRate - spreadRate);
      if(crownRate >= crownActiveRate)
         maxRate = max(crownRate, maxRate);

   }
   return maxRate * 3.28 / 60.0;
}

/*
Function: testCrownRate
Input: TBD
Purpose: tests crown rate in each update step
*/
float FireSim::Clamp(float val, float flr, float ceiling){
   if(val >= flr && val <= ceiling){
      return val;
   }
   if(val < flr){
      return flr;
   }
   return ceiling;
}


/*
Function: setSimSize
Input: height,width of grid for testing
Purpose: allows size to be set for generation of data in simulation tests
*/
void FireSim::SetSimSize(int x, int y){
   sim_dim_x_ = x;
   sim_dim_y_ = y;
}

/*
Function: burnDistance
Input: distance, rate, timestep 
Purpose: reduce distance for burning over several timesteps
*/
bool FireSim::BurnDistance(float &dist, float rate, float step){
    bool torched = false;
    // lower distance based on roth rate
        // t = d / r;
        // d = d - r * timeStep
    dist = dist - rate * step;
    if( dist < 0){
      // dist = -1*dist;
      dist = 0;
      torched = true; 
    }
    return torched;
}


/*
Function: accelerate
Input: TBD
Purpose: run acceleration code
*/
template<typename T>
T* GISToFloatArray(char* fname, int interpWidth, int interpHeight)
{
  // Important note ------ Gdal considers images to be north up
  // the origin of datasets takes place in the upper-left or North-West corner.
  // Now to create a GDAL dataset
  // auto ds = ((GDALDataset*) GDALOpen(fname,GA_ReadOnly));
  GDALDataset* ds = ((GDALDataset*) GDALOpen(fname,GA_ReadOnly));
  if(ds == NULL)
  {
    return NULL;
  }
  
  // Creating a Raster band variable
  // A band represents one whole dataset within a dataset
  // in your case your files have one band.
  GDALRasterBand  *poBand;
  int             nBlockXSize, nBlockYSize;
  int             bGotMin, bGotMax;
  double          adfMinMax[2];
  
  // Assign the band      
  poBand = ds->GetRasterBand( 1 );
  poBand->GetBlockSize( &nBlockXSize, &nBlockYSize );

  // find the min and max
  adfMinMax[0] = poBand->GetMinimum( &bGotMin );
  adfMinMax[1] = poBand->GetMaximum( &bGotMax );
  if( ! (bGotMin && bGotMax) )
    GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE, adfMinMax);
  int min = adfMinMax[0];
  int max = adfMinMax[1];

  // get the width and height of the band or dataset
  int width = poBand->GetXSize();
  int height = poBand->GetYSize();

  // GDAL can handle files that have multiple datasets jammed witin it
  int bands = ds->GetRasterCount();

  // the float variable to hold the DEM!
  T *pafScanline;
  // std::cout << "Min: " << adfMinMax[0] << " Max: " << adfMinMax[1] << endl;
  int dsize = 256;
  // pafScanline = (T *) CPLMalloc(sizeof(T)*width*height);
  pafScanline = (T *) CPLMalloc(sizeof(T)*interpWidth*interpHeight);

  // Lets acquire the data.  ..... this funciton will interpolate for you
  // poBand->RasterIO(GF_Read,0,0,width,height,pafScanline,width,height,GDT_Float32,0,0);
  poBand->RasterIO(GF_Read,0,0,width,height,pafScanline,interpWidth,interpHeight,GDT_Float32,0,0);
  //        chage these two to interpolate automatically ^      ^

  // The Geotransform gives information on where a dataset is located in the world
  // and the resolution.
  // for more information look at http://www.gdal.org/gdal_datamodel.html
  double geot[6];
  ds->GetGeoTransform(geot);

  // Get the x resolution per pixel(south and west) and y resolution per pixel (north and south)
  // float xres = geot[1];
  // float yres = geot[5];
  // string proj;
  // proj = string(ds->GetProjectionRef());

  // You can get the projection string
  // The projection gives information about the coordinate system a dataset is in
  // This is important to allow other GIS softwares to place datasets into the same
  // coordinate space 
  // char* test = &proj[0];

  // The origin of the dataset 
  // float startx = geot[0]; // east - west coord.
  // float starty = geot[3]; // north - south coord.

  
  // here is some code that I used to push that 1D array into a 2D array
  // I believe this puts everything in the correct order....
  /*for(int i = 0; i < hsize; i++)
  {
    for(int j = 0; j < wsize; j++)
    {
      //cout << i << j << endl << pafS;
      vecs[i][j] = pafScanline[((int)i)*(int)wsize+((int)j)];
      if(vecs[i][j]>0 && vecs[i][j] > max)
      {
          max = vecs[i][j];
      }
      if(vecs[i][j]>0 && vecs[i][j] < min)
      {
          min = vecs[i][j];
      }
    }
   }*/
   //CPLFree(pafScanline);
   return pafScanline;

}