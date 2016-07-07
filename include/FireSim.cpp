//
// Created by jsmith on 10/2/15.
//

#include "FireSim.h"
#include <algorithm>
#include <cmath>
//#include <stdio.h>
//#include <unistd.h>
//#define GetCurrentDir getcwd

const int INF = 32767;

#define b printf("%u\n", __LINE__);


float* GISToFloatArray(const char*, int, int, float&);
int* GISToIntArray(const char*, int, int, float&);

/*
Constructor: builds simplest test case for testing code
*/
FireSim::FireSim(int _x /*cols, width*/, int _y/*rows, height*/,
                 std::string fuel_model_name, std::string fuel_moistures_name){
   printf("Initializing Simulation to Test Setting\n");

   // declare 2d map data
   sim_dim_x_ = _x;
   sim_dim_y_ = _y;


   root_path_ = "/home/jessie/Documents/simulator/";

   std::string tmp = "data/default.fmd";
   tmp = root_path_ + tmp;
   std::cout << tmp << std::endl;
   const char * fname_tmp = fuel_model_name.c_str();
   _models = sim::readFuelModels(fname_tmp);

   tmp = "data/kyle.fms";
   tmp = root_path_ + tmp;
   std::cout << tmp << std::endl;
   fname_tmp = fuel_moistures_name.c_str();
   _moistures = sim::readFuelMoistures(fname_tmp);
   int numModels = _models.size();
   int numMoistModels = _moistures.size();
   foliar_moisture = 1.0f;

   toa_ = new float*[sim_dim_y_];
   roth_data_ = new vec4*[sim_dim_y_];
   current_ros_ = new float*[sim_dim_y_];
   original_toa_ = new float*[sim_dim_y_];
   update_stamp_ = new int*[sim_dim_y_];
   source_data_texture_ = new point*[sim_dim_y_];

   // rothermel vals
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

   wind_t_ = new vec2*[sim_dim_y_];
   dead_moistures_t_ = new vec3[numMoistModels];
   live_moistures_t_ = new vec2[numMoistModels];


   for(int i = 0; i < sim_dim_y_; i++){
      toa_[i] = new float[sim_dim_x_];
      roth_data_[i] = new vec4[sim_dim_x_];
      current_ros_[i] = new float[sim_dim_x_];
      original_toa_[i] = new float[sim_dim_x_];
      update_stamp_[i] = new int[sim_dim_x_];
      source_data_texture_[i] = new point[sim_dim_x_];

      // rothermel
      wind_t_[i] = new vec2[sim_dim_x_];
   }

   acceleration_constant_ = 1.0;

   // Simulation Data Members
   ign_time_ = new float[sim_dim_x_ * sim_dim_y_];
   ign_time_new_ = new float[sim_dim_x_ * sim_dim_y_];
   burn_dist_ = new float*[sim_dim_x_ * sim_dim_y_];
   for(int i = 0; i < sim_dim_x_ * sim_dim_y_; i++){
      burn_dist_[i] = new float[8];
   }

//   cell_size_ = 200;
   l_n_ = new float[16];

   // Spotting Constants
   C_d = 1.2f;
   p_a = 0.0012;
   p_s = 0.3f;
   g = 9.8f;
   K = 0.0064;
   a_x = 5.963;
   b_x = 4.563;
//   D_p = 0.01; // 10mm diameter, just a random value I set
   D_p = 0.01; // 10mm diameter, just a random value I set
   B = 40;
   v_o = (float) pow((M_PI*g*p_s*D_p)/(2*C_d*p_a),0.5f);
   tau = (float) ((4*C_d*v_o)/(K*M_PI*g));
//   std::cout << v_o << ' ' << tau << std::endl;
//    std::cout << "constructed" << std::endl;
}

/*
Destructor: builds simplest test case for testing code
*/
FireSim::~FireSim(){
   // delete all memory

   // blah blah
   // std::std::cout << "Deallocating memory" << std::std::endl;
   // sim_dim_x_ = sizeX;
   // sim_dim_y_ = sizeY;
   for(int i = 0; i < sim_dim_y_; i++){
      delete toa_[i];
      delete roth_data_[i];
      delete current_ros_[i];
      delete original_toa_[i];
      delete update_stamp_[i];
      delete source_data_texture_[i];

      // rothermel
      delete wind_t_[i];
   }

   delete toa_;
   delete roth_data_;
   delete current_ros_;
   delete original_toa_;
   delete update_stamp_;
   delete source_data_texture_;

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


   acceleration_constant_ = 1.0;

   // std::cout << "test" << std::endl;
   // Simulation Data Members
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
void FireSim::Init(std::string fuel_file, std::string terrain_file,
                   std::string canopy_height_file, std::string crown_base_height_file,
                   std::string crown_bulk_density_file, float wind_x, float wind_y){
   // read from files:
   int cell = 0;
   float junk;
   float* slopeTexTmp = NULL;
   GDALAllRegister();
   const char * fname_tmp = fuel_file.c_str();
   fuel_t_ = GISToIntArray(fname_tmp, sim_dim_x_, sim_dim_y_, junk);

   fname_tmp = terrain_file.c_str();
   slopeTexTmp = GISToFloatArray(fname_tmp, sim_dim_x_ *3, sim_dim_y_ *3, cell_size_);

   fname_tmp = canopy_height_file.c_str();
   canopy_height_ = GISToFloatArray(fname_tmp, sim_dim_x_, sim_dim_y_, junk);

   // Crown base height data is only used for calculating I_o so it doesn't need to be stored separately
   fname_tmp = crown_base_height_file.c_str();
   I_o_ = GISToFloatArray(fname_tmp, sim_dim_x_, sim_dim_y_, junk);
   for(int i = 0; i < sim_dim_x_*sim_dim_y_; i++){
      I_o_[i] = pow(0.01 * I_o_[i] * (460.0 + 25.9 * foliar_moisture), 1.5);
   }

   // Crown Bulk Density data is only used for calculating RAC so it doesn't need to be stored separately
   fname_tmp = crown_bulk_density_file.c_str();
   RAC_ = GISToFloatArray(fname_tmp, sim_dim_x_, sim_dim_y_, junk);
   for(int i = 0; i < sim_dim_x_*sim_dim_y_; i++){
      RAC_[i] = 3.0f / RAC_[i];
   }

   for(int i = 0; i < sim_dim_y_; i++){
      for(int j = 0; j < sim_dim_x_; j++, cell++){
         toa_[i][j] = 20.f;
         roth_data_[i][j].x = roth_data_[i][j].y = roth_data_[i][j].z = 0.f;
         current_ros_[i][j] = 0.f;
         // fuel_t_[i][j] = 0.;
         original_toa_[i][j] = 20.f;
         update_stamp_[i][j] = 0;
         source_data_texture_[i][j].x = source_data_texture_[i][j].y = 0.f;

         // Rothermel Data Members
         wind_t_[i][j].x = wind_x;
         wind_t_[i][j].y = wind_y;

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

   time_step_ = 2.0;


   int i = 0;
   for (std::vector<sim::FuelModel>::iterator it = _models.begin();
        it != _models.end(); it++)
   {
//      std::cout << it->modelNumber << std::endl;
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
      // dead_sav_burnable_b_[i].w = it->burnable? 100.0f : 0.0f;
      dead_sav_burnable_b_[i].w = 100.0f;

      fuel_sav_accel_b_[i].x = it->fuelSAV;
      fuel_sav_accel_b_[i].y = it->accelerationConstant;
      i++;
   }
//   for (std::map<int,int>::iterator it=fuel_model_dict_.begin(); it!=fuel_model_dict_.end(); ++it)
//      std::cout << it->first << " => " << it->second << '\n';

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

   float orthoSize = cell_size_;
   float diagSize = (float) (cell_size_ * sqrt(2));
   float superSize = (float) sqrt(pow(cell_size_, 2) + pow(cell_size_ * 2, 2));
   static float L_n_tmp[16] =  { orthoSize, diagSize, orthoSize, diagSize, orthoSize, diagSize,
                                 orthoSize, diagSize, superSize,superSize,superSize,superSize,
                                 superSize,superSize,superSize,superSize};
   for(int j = 0; j < 16; j++){
      l_n_[j] = L_n_tmp[j];
   }
}

/*
Function: UpdateSpreadData
Input: The necessary inputs are the values that are found in the textures/buffers
       in the FuelModel.h/Simulator.cpp files in Roger's code
Shader base: rothermel
Purpose: This runs rothermel's equations to initialize simulation
*/
void FireSim::UpdateSpreadData(){
   // This is where rothermel's shader is implemented

   std::cout << "Updating Spread Data . . ." << std::endl;
   int cell = 0;   /* row, col, and index of neighbor cell */
   float dist = 10.f;
   int counter = 0;

   for(int i = 0; i < sim_dim_y_; i++){
      for(int j = 0; j < sim_dim_x_; j++, cell++){
         // Shader code: int fuelModel = texture2D(fuel_t_, gl_TexCoord[1].st).r;
         // gl_TexCoord[1].st corresponds to fuel_t_.xy
          int fuelModel = fuel_t_[cell];
//         fuelModel = 1;

         if(fuelModel == 99){
            roth_data_[i][j].w =1.0;
            // gl_FragData[0] = vec4(maxSpreadRate,
            //                       ellipseEccentricity, spreadDirection, intensityModifier);

            roth_data_[i][j].x = 0.0f;
            // std::cout << maxSpreadRate;
            roth_data_[i][j].y = 1.0f;
            // std::cout << spreadDirection;
            roth_data_[i][j].z = 1.0f;
            continue;
         }
//          std::cout << "FUEL MODEL " << fuelModel << std::endl;

         // FOR TESTING: Must fix interpolation of fuelModel data
         // TO FIX: Build dictionary of fuel model data
//         int fuelModel = 10;

         vec4 dead1h, deadSAVBurnable, dead10h, dead100h, liveH,
               liveW, fineDeadExtinctionsDensity, areasReactionFactors,
               slopeWindFactors, residenceFluxLiveSAV;
         vec2 fuelSAVAccel;

         vec3 slopeAspectElevation;
         vec2 wind;
         vec3 deadMoistures;
         vec2 liveMoistures;

         // Get data into vars
//          std::cout << dead_sav_burnable_b_[fuelModel].x << " " <<
//                  dead_sav_burnable_b_[fuelModel].y << " " <<
//                  dead_sav_burnable_b_[fuelModel].z << " " <<
//                  dead_sav_burnable_b_[fuelModel].w << std::endl;
         deadSAVBurnable = dead_sav_burnable_b_[fuelModel];
         // std::cout << deadSAVBurnable.x << " " <<
         //         deadSAVBurnable.y << " " <<
         //         deadSAVBurnable.z << " " <<
         //         deadSAVBurnable.w << std::endl;
         if(deadSAVBurnable.w < 50.0){
            std::cout << "Warning: Something may have gone wrong. Check that Files were read Correctly." << std::endl;
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

         float maxSpreadRate = 0.f;
         float ellipseEccentricity = 0.f;
         float spreadDirection = 0.f;
         float spreadModifier = 0.f;
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
         //    std::cout << fuelMoistures[c] << std::endl;
         // }

         float liveExtinction = 0.0;
         if(liveH.y > 0.0 || liveW.y > 0.0){
            float fineDeadMoisture = 0.0;
            if (fineDeadExtinctionsDensity.x > 0.0)
               fineDeadMoisture = weightedFuelModel / fineDeadExtinctionsDensity.x;

            liveExtinction =
                  (fineDeadExtinctionsDensity.z *
                   (1.0f - fineDeadMoisture / fineDeadExtinctionsDensity.y)) - 0.226f;
            liveExtinction = std::max(liveExtinction, fineDeadExtinctionsDensity.y);
         }

         float heatOfIgnition =
               areasReactionFactors.x *
               ((250.0f + 1116.0f * fuelMoistures[0]) * dead1h.z * dead1h.x +
                (250.0f + 1116.0f * fuelMoistures[1]) * dead10h.z * dead10h.x +
                (250.0f + 1116.0f * fuelMoistures[2]) * dead100h.z * dead100h.x) +
               areasReactionFactors.y *
               ((250.0f + 1116.0f * fuelMoistures[3]) * liveH.z * liveH.x +
                (250.0f + 1116.0f * fuelMoistures[4]) * liveW.z * liveW.x);
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
         // std::cout << "baseSpreadRate" << baseSpreadRate << std::endl;

         float slopeFactor = slopeWindFactors.x * slopeAspectElevation.x * slopeAspectElevation.x;
         float windFactor = 0.0;
         if (wind.x > 0.0)
            windFactor = slopeWindFactors.y * pow(wind.x, slopeWindFactors.z);

         spreadModifier = slopeFactor + windFactor;
         // std::cout << slopeFactor << " " << windFactor << std::endl;
         float upslope;
         if (slopeAspectElevation.y >= 180.0)
            upslope = slopeAspectElevation.y - 180.0f;
         else
            upslope = slopeAspectElevation.y + 180.0f;

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
            if (upslope <= wind.y)https://play.google.com/store/books/details?id=rlOgae6p898C&rdid=book-rlOgae6p898C&rdot=1
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
            // std::cout << "spreadmoid: " << spreadModifier << std::endl;
            if (spreadModifier > 0.0)
               updateEffectiveWindspeed = 1;
            checkEffectiveWindspeed = 1;

            float addedAngle = 0.0;
            if (addedSpeed > 0.0)
               addedAngle = asin(Clamp(fabs(y) / addedSpeed, -1.0, 1.0));
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
         //maxSpreadRate = maxSpreadRate * (1.0 - exp(-acceleration_constant_ * burnTime / 60.0));
         //float firelineIntensity =
         // 3.4613 * (384.0 * (reactionIntensity / 0.189275)) *
         //    (maxSpreadRate * 0.30480060960) / (60.0 * fuelSAV);
         //firelineIntensity =
         // reactionIntensity * 12.6 * maxSpreadRate / (60.0 * fuelSAV);
//         intensity_modifier[i][j] =
//               3.4613 * (384.0 * (reactionIntensity / 0.189275)) *
//               (0.30480060960) / (60.0 * fuelSAV);
         roth_data_[i][j].w =
               3.4613 * (384.0 * (reactionIntensity / 0.189275)) *
               (0.30480060960) / (60.0 * fuelSAV);
         // gl_FragData[0] = vec4(maxSpreadRate,
         //                       ellipseEccentricity, spreadDirection, intensityModifier);

         roth_data_[i][j].x = maxSpreadRate;
         // std::cout << maxSpreadRate;
         roth_data_[i][j].y = spreadDirection;
         // std::cout << spreadDirection;
         roth_data_[i][j].z = ellipseEccentricity;
         // std::cout << ellipseEccentricity << std::endl;

      }
   }
    std::cout << "Spread Data Calculated" << std::endl;
}


/*
Function: TestCrownRate
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
Function: BurnDistance
Input: distance, rate, timestep
Purpose: reduce distance for burning over several timesteps
*/
bool FireSim::BurnDistance(float &dist, float rate, float step){
   bool torched = false;
   // lower distance based on roth rate
   // t = d / r;
   // d = d - r * time_step_
   dist = dist - rate * step;
   if( dist < 0){
      dist *= -1;
//      dist = 0;
      torched = true;
   }
   return torched;
}





float* GISToFloatArray(const char* fname, int interpWidth, int interpHeight, float &cell_size)
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
   float *pafScanline;
   // std::std::cout << "Min: " << adfMinMax[0] << " Max: " << adfMinMax[1] << std::endl;
   int dsize = 256;
   // pafScanline = (T *) CPLMalloc(sizeof(T)*width*height);
   pafScanline = (float*) CPLMalloc(sizeof(float)*interpWidth*interpHeight);

   // Lets acquire the data.  ..... this funciton will interpolate for you
   // poBand->RasterIO(GF_Read,0,0,width,height,pafScanline,width,height,GDT_Float32,0,0);
   poBand->RasterIO(GF_Read,0,0,width,height,pafScanline,interpWidth,interpHeight,GDT_Int32,0,0);
   //        chage these two to interpolate automatically ^      ^

   // The Geotransform gives information on where a dataset is located in the world
   // and the resolution.
   // for more information look at http://www.gdal.org/gdal_datamodel.html
   double geot[6];
   ds->GetGeoTransform(geot);

   // Get the x resolution per pixel(south and west) and y resolution per pixel (north and south)
   float xres = geot[1];
   float yres = geot[5];
   // Calc Cell Size
   cell_size = (xres * width) / interpWidth;
//   std::cout << "X RES: " << xres << " Y RES: " << yres << std::endl;
//   std::cout << width << ' ' << height << std::endl;
   // std::string proj;
   // proj = std::string(ds->GetProjectionRef());

   // You can get the projection std::string
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
       //std::cout << i << j << std::endl << pafS;
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


int* GISToIntArray(const char* fname, int interpWidth, int interpHeight, float &cell_size)
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
   int *pafScanline;
   // std::std::cout << "Min: " << adfMinMax[0] << " Max: " << adfMinMax[1] << std::endl;
   int dsize = 256;
   // pafScanline = (T *) CPLMalloc(sizeof(T)*width*height);
   pafScanline = (int *) CPLMalloc(sizeof(int)*interpWidth*interpHeight);

   // Lets acquire the data.  ..... this funciton will interpolate for you
   // poBand->RasterIO(GF_Read,0,0,width,height,pafScanline,width,height,GDT_Float32,0,0);
   poBand->RasterIO(GF_Read,0,0,width,height,pafScanline,interpWidth,interpHeight,GDT_Int32,0,0);
   //        chage these two to interpolate automatically ^      ^

   // The Geotransform gives information on where a dataset is located in the world
   // and the resolution.
   // for more information look at http://www.gdal.org/gdal_datamodel.html
   double geot[6];
   ds->GetGeoTransform(geot);

   // Get the x resolution per pixel(south and west) and y resolution per pixel (north and south)
   float xres = geot[1];
   float yres = geot[5];
   // Calc Cell Size
   cell_size = (xres * width) / interpWidth;

   return pafScanline;

}


bool FireSim::SetFire(int _x, int _y, int _time){
   if(sim_dim_x_ <= 0 || sim_dim_y_ <=0){
      printf("Error: You cannot set fire when simulation not initialized\n");
      return false;
   }
   else{
      int ignSpot = _x * sim_dim_x_ + _y;
      ign_time_[ignSpot] = _time;
      ign_time_new_[ignSpot] = _time;
   }

}