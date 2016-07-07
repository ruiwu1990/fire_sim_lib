//
// Created by jsmith on 10/26/15.
//

#include "SEQ.h"

const int INF = 32767;

SequentialSpread::SequentialSpread(int _x, int _y) {
   simulation_ = new FireSim(_x, _y, "/home/jessie/Documents/simulator/data/default.fmd",
                             "/home/jessie/Documents/simulator/data/default.fmd/data/kyle.fms");
   simulation_->Init("/home/jessie/Documents/simulator/data/fixed.fuel",
                     "/home/jessie/Documents/simulator/data/fixed.dem",
                     "/home/jessie/Documents/simulator/data/canopy_ht.asc",
                     "/home/jessie/Documents/simulator/data/crown_base_ht.asc",
                     "/home/jessie/Documents/simulator/data/crown_bulk_density.asc", 0,0);
   simulation_->UpdateSpreadData();
}

SequentialSpread::~SequentialSpread() {
   simulation_ = NULL;
   free(maxspreadrate_);
   free(curspreadrate_);
}

bool SequentialSpread::Init() {
   sim_size_ = simulation_->sim_dim_x_ * simulation_->sim_dim_y_;
   sim_rows_ = simulation_->sim_dim_y_;
   sim_cols_ = simulation_->sim_dim_x_;
   maxspreadrate_ = (float*) malloc(sim_size_*16*sizeof(float));
   curspreadrate_ = (float*) malloc(sim_size_*16*sizeof(float));
   ember_map_ = (bool*) malloc(sim_size_*16*sizeof(bool));
   test_map_ = (float*) malloc(sim_size_*16*sizeof(float));
   return true;
}


/*
Function: RunSimulationBD
Input: None
Purpose: Runs the BD simulation
*/
bool SequentialSpread::RunSimulationBD(int step_size) {

   int cell, row, col, nrow, ncol, ncell;
   int termFlag = 0;
   int ignspot = simulation_->sim_dim_x_*simulation_->sim_dim_y_ / 2 + simulation_->sim_dim_y_ /2;

   struct timeval start, fin;
   float pi = 3.14159;
   float ROS = 0.f;
//   float superSize = sqrt(pow(simulation_->cell_size_, 2) + pow(simulation_->cell_size_*2, 2));
   /* neighbor's address*/     /* N  NE   E  SE   S  SW   W  NW  NNW NNE NEE SEE SSE SSW SWW NWW*/
   static int nCol[16] =        {  0,  1,  1,  1,  0, -1, -1, -1, -1, 1, 2, 2, 1, -1, -2, -2};
   static int nRow[16] =        {  1,  1,  0, -1, -1, -1,  0,  1, 2, 2, 1, -1, -2, -2, -1, 1};

   std::cout << "Beginning Simulation (Burning Distances)" << std::endl;
   int corner = 0;
   int counter = 0;
   char simType[20];
//   gettimeofday(&start, NULL);
//   bool corner_one = true;
//   bool corner_two = true;
//   bool corner_three = true;
//   bool corner_four = true;
   float t = 0.0;
//   while((corner < 4 or termFlag < 15) and t < INF){
   while(counter < step_size){
//   while(corner_one || corner_two || corner_three || corner_four){
      for ( cell=0, row=0; row < sim_rows_; row++ ){
         for ( col=0; col < sim_cols_; col++, cell++ ){
            // check not "ignited"
            if(simulation_->ign_time_[cell] == INF){
               continue;
            }
            // check neighbors for ignition
            for(int n = 0; n < 8; n++){
               nrow = row + nRow[n];
               ncol = col + nCol[n];
               if ( nrow<0 || nrow>=sim_rows_ || ncol<0 || ncol>=sim_cols_ )
                  continue;
               ncell = ncol + nrow*sim_cols_;

               // check for already lit
               if(simulation_->ign_time_[ncell] < INF){
                  continue;
               }
               // Calc roth values
               ROS = curspreadrate_[ncell*16 + n];

               // Burn distance
               bool test = simulation_->BurnDistance(simulation_->burn_dist_[ncell][n],
                                            ROS,
                                            simulation_->time_step_);
               // Propogate fire step:
               if(test){
                  // Do more accurate time calc
                  float step_time = simulation_->burn_dist_[ncell][n] / ROS;
                  simulation_->ign_time_new_[ncell] = t + step_time;

                  // Inherit fire rate props from neighbor
                  curspreadrate_[ncell] = ROS;

//                  if(nrow == (sim_rows_ - 1) && ncol == (sim_cols_ - 1)){
//                     corner += 1;
////                     cout << "corner 1\n";
////                     corner_one = false;
//                  }
//                  if(nrow == 0 && ncol == (sim_cols_ - 1)){
//                     corner += 1;
////                     cout << "corner 2\n";
////                     corner_two = false;
//                  }
//                  if(nrow == 0 && ncol == 0){
//                     corner += 1;
////                     cout << "corner 3" << endl;
////                     corner_three = false;
//                  }
//                  if(nrow == (simulation_->sim_dim_x_ - 1) && ncol == 0){
//                     corner += 1;
////                     cout << "corner 4\n";
////                     corner_four = false;
//                  }
               }
            }
         }
      }
      for(int i = 0; i < sim_size_; i++){
         if(simulation_->ign_time_new_[i] < INF){
            simulation_->ign_time_[i] = simulation_->ign_time_new_[i];
            simulation_->ign_time_new_[i] = INF;
         }
      }
//      if(corner == 4)
//         break;
      if(corner >= 4) {
         termFlag++;
      }
      t+= simulation_->time_step_;
      counter++;
//      std::cout << t << std::endl;
      // Test for crowning
      TestCrownRate(t);
      // Test for spotting
      TestSpotting(t);
      // Accelerate Fire
      Accelerate();
//         counter++;
   }

   std::cout << "End of Simulation" << std::endl << std::endl;

   return true;
}


/*
Function: RunSimulationIMT
Input: None
Purpose: Runs the IMT simulation
*/
bool SequentialSpread::RunSimulationIMT(int step_size){
   std::cout << "Beginning Simulation (Iterative Minimal Time)" << std::endl;
   float angles[16] = {  90,  45,  0, -45, -90, -115,  180,  115, 63.4349, 26.5651, -26.5651f,
                         -63.4349f, -116.5651f, -153.4349f, 153.4349f, 116.5651f};
   float ignCell = 0.f;
   float ignCellNew = 0.f;
   float ignTimeMin = INF;
   int simCounter = 0;
   int cell, row, col, nrow, ncol, ncell, counter = 0;
   float ROS, ROS_Update;
   static int nCol[16] =        {  0,  1,  1,  1,  0, -1, -1, -1, -1, 1, 2, 2, 1, -1, -2, -2};
   static int nRow[16] =        {  1,  1,  0, -1, -1, -1,  0,  1, 2, 2, 1, -1, -2, -2, -1, 1};
   bool* check = new bool[sim_size_];
   for(int z = 0; z < sim_size_; z++){
      check[z] = false;
   }

   // int counter = 0;
   char simType[20];
//   while(simCounter < sim_size_){
   while(counter < step_size){
       counter++;

      // Loop through all cells
      // for(cell = 0; cell < CELLS; cell++){
      for ( cell=0, row=0; row< sim_rows_; row++ ){
         for ( col=0; col< sim_cols_; col++, cell++ ){
            if(check[cell] == true)
               continue;

            // Check for simulation completion
            ignCell = simulation_->ign_time_[cell];
            ignCellNew = simulation_->ign_time_new_[cell];

            if(fabs(ignCell - ignCellNew) < .00001 && ignCell != INF
               && ignCellNew != INF && check[cell] != true){
               simCounter++;
               check[cell] = true;
               continue;
            }

            if(ignCell > 0){
               ignTimeMin = INF;
               // Loop through neighbors
               for(int n = 0; n < 16; n++){
                  // find neighbor cell index
                  nrow = row + nRow[n];
                  ncol = col + nCol[n];
                  // std::cout << row << ' ' << col << ' ' << std::endl;
                  if ( nrow<0 || nrow>= sim_rows_ || ncol<0 || ncol>=  sim_cols_ ) {
                     continue;
                  }
                  ncell = ncol + nrow*sim_cols_;

                  ROS = curspreadrate_[ncell * 16 + n];

                  float ignTimeNew = simulation_->ign_time_[ncell] + (simulation_->l_n_[n] / ROS);// * 100;
                  ignTimeMin = ignTimeNew*(ignTimeNew < ignTimeMin) + ignTimeMin*(ignTimeNew >= ignTimeMin);
                  ROS_Update = ROS*(ignTimeNew < ignTimeMin) + curspreadrate_[cell]*(ignTimeNew >= ignTimeMin);
               }
               simulation_->ign_time_new_[cell] = (int)ignTimeMin;

               // Inherit fire rate properties from neighbor
               curspreadrate_[cell] = ROS_Update;

            }
         }
      }

      // Swap pointers to loop
      float *temp = simulation_->ign_time_;
      simulation_->ign_time_ = simulation_->ign_time_new_;
      simulation_->ign_time_new_ = temp;

      // Perform Crowning test
      TestCrownRate(0.0);

      // Perform Fire Acceleration
      Accelerate();
   }
   std::cout << "End of Simulation" << std::endl;
}


/*
Function: RunSimulationMT
Input: None
Purpose: Runs the MT simulation
*/
bool SequentialSpread::RunSimulationMT(int step_size) {
   std::cout << "Beginning Simulation (Minimal Time)" << std::endl;
   int cell, row, col, nrow, ncol, ncell;
   float ROS;
   static int nCol[16] =        {  0,  1,  1,  1,  0, -1, -1, -1, -1, 1, 2, 2, 1, -1, -2, -2};
   static int nRow[16] =        {  1,  1,  0, -1, -1, -1,  0,  1, 2, 2, 1, -1, -2, -2, -1, 1};
   float angles[16] =           {  90,  45,  0, -45, -90, -115,  180,  115, 63.4349, 26.5651, -26.5651f,
                                   -63.4349f, -116.5651f, -153.4349f, 153.4349f, 116.5651f};
   int counter = 0;
   char simType[20];
   // Initialize time_next for this simulation
   simulation_->time_next_ = 0;
//   while(simulation_->time_next_ < INF){
   while(counter < step_size){
      simulation_->time_now_ = simulation_->time_next_;
      simulation_->time_next_ = INF;
      counter++;
      TestCrownRate(0.0);
      // Accelerate Fire
      Accelerate();

      // Loop through all cells
      for ( cell=0, row=0; row< sim_rows_; row++ ){
         for ( col=0; col< sim_cols_; col++, cell++ ){
            // printf("Cell: %d\n", cell);
            if(simulation_->time_next_ > simulation_->ign_time_[cell] && simulation_->ign_time_[cell] > simulation_->time_now_){
               simulation_->time_next_ = simulation_->ign_time_[cell];
               // printf("Hitting here: %d \n", cell);
            }
            else if( simulation_->ign_time_[cell] == simulation_->time_now_){
               for(int n = 0; n < 16; n++){
                  // for(int n = 0; n < 8; n++){
                  // find neighbor cell index
                  nrow = row + nRow[n];
                  ncol = col + nCol[n];
                  // std::cout << row << ' ' << col << ' ' << std::endl;
                  if ( nrow<0 || nrow>= sim_rows_ || ncol<0 || ncol>=  sim_cols_ ) {
                     continue;
                  }
                  ncell = ncol + nrow*sim_cols_;

                  // If neighbor is unburned
                  if(simulation_->ign_time_[ncell] > simulation_->time_now_){
                      // compute ignition time
                     ROS = curspreadrate_[cell * 16 + n];

                     float ignTimeNew = simulation_->time_now_ + (simulation_->l_n_[n] / ROS);// * 100;

                     if(ignTimeNew < simulation_->ign_time_[ncell]){
                        simulation_->ign_time_[ncell] = (int)ignTimeNew;
                        curspreadrate_[ncell] = ROS;
                     }
                     if(ignTimeNew < simulation_->time_next_){
                        simulation_->time_next_ = (int)ignTimeNew;
                     }
                  }
               }
            }
         }
      }
   }
   std::cout << "End of Simulation" << std::endl;
}

/*
Function: WriteToFile
Input: None
Purpose: Writes the results of the simulation to a file in the out/ folder
*/
bool SequentialSpread::WriteToFile() {
   std::ofstream fout;
   std::string filename;
   filename += simulation_->root_path_;
   filename += "out/SEQ_test.csv";
   fout.open(filename.c_str());
   std::cout << "Writing Results To: " << filename << std::endl;
   for(unsigned int i = 0; i < sim_size_; i++){
      if(i % simulation_->sim_dim_x_ == 0 && i !=0){
         fout << '\n';
      }
      fout << (int) simulation_->ign_time_[i] << ",";
   }
   fout.close();
//   for(int i = 0; i < sim_size_*16; i++){
//      if(i%(simulation_->sim_dim_x_*16) == 0 && i != 0){
//         std::cout << std::endl;
//      }
//      std::cout << test_map_[i];
//   }
   return true;
}


/*
Function: CalcMaxSpreadRates
Input: None
Purpose: Calculate the maximum spread rates once so the GPU doesn't have to do it
         on every iteration. It also intializes the current spread rate to 0.
*/
bool SequentialSpread::CalcMaxSpreadRates() {
   /*
      cells formatting will be as follows:
      cell[x*16] is reference cell, following 8/16 vals are direction data
      N  NE   E  SE   S  SW   W  NW  NNW NNE NEE SEE SSE SSW SWW NWW
    */
   int dirs = 16;
   int cell = 0;
   float angles[16] = {  90,  45,  0, -45, -90, -115,  180,  115, 63.4349, 26.5651, -26.5651f,
                         -63.4349f, -116.5651f, -153.4349f, 153.4349f, 116.5651f};
   for(int row = 0; row < sim_rows_; row++) {
      for (int col = 0; col < sim_cols_; col++) {
         for (int i = 0; i < dirs; i++, cell++) {
            maxspreadrate_[cell] = (float) (simulation_->roth_data_[row][col].x * (1.0f - simulation_->roth_data_[row][col].z) /
                                            (1.0f - simulation_->roth_data_[row][col].z *
                                                    cos( simulation_->roth_data_[row][col].y *
                                                               3.14159f / 180.f - angles[i])));

            curspreadrate_[cell] = 0.f;
            ember_map_[cell] = false;
            test_map_[cell] = 0.f;
         }
      }
   }
}



/*
Function: Accelerate
Input: None
Purpose: run acceleration code. This code will calculate the rate at which the
         fire will spread will increase and manage the increase of the rate
*/
bool SequentialSpread::Accelerate() {
   float current, ratio, timetomax;
   int cell = 0;
   int dirs = 16;

   float acceleration_constant = simulation_->fuel_sav_accel_b_[1].y;//////////////////////////////////////// THIS NEEDS TO BE FUEL MODEL AT CELL I
//   cout << simulation_->fuel_sav_accel_b_[1].y << endl;
   // Accelerate every cell
   for(int row = 0; row < sim_rows_; row++) {
      for (int col = 0; col < sim_cols_; col++) {
         for (int i = 0; i < dirs; i++, cell++) {
            current = std::min(curspreadrate_[cell], maxspreadrate_[cell]);
            ratio = current / maxspreadrate_[cell];
            timetomax = -log(1.0f - ratio) / acceleration_constant;
            curspreadrate_[cell] = simulation_->Clamp(simulation_->time_step_ - ratio, 0.0f,1.0f) * (maxspreadrate_[cell] - current) + current;
         }
      }
   }

}



/*
Function: TestCrownRate
Input: None
Purpose: This tests if the fire has crowned, then if the fire is active, and adjusts the spread rate accordingly
*/
bool SequentialSpread::TestCrownRate(float t){
   float t_t, t_0, t_1, t_2, t_3, z_F, I_b, z_o, velocity_ratio, r;
   float max;
   int row, col;
   for(int cell = 0, cell_sr = 0; cell < sim_size_; cell++, cell_sr+= 16) {
//      row = cell / sim_rows_;
//      if(row == 0)
//         col = cell;
//      else
////         col = (cell - sim_rows_) / row;
//         col = cell - row*sim_cols_;

      row = cell / sim_cols_;
      col = cell % sim_cols_;

      if (simulation_->ign_time_[cell] > t || simulation_->canopy_height_[cell] <= 0.f) {
//         std::cout << "continuing: " << cell << ' ' << t << std::endl;
         continue;
      }
//      std::cout << row << " " << col << std::endl;
      I_b = curspreadrate_[cell_sr] * simulation_->roth_data_[row][col].w; // firelineIntensity is I_b
      if (I_b > simulation_->I_o_[cell]) {
         // START SPOTTING CODE //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
         // Launch Ember for Spotting
         if(!ember_map_[cell_sr]){ // ember has not launched => launch ember
//            z_F = (float) (0.0775 * pow(I_b,0.46));
            z_F = (float) (0.0775 * pow(I_b,0.46) + simulation_->canopy_height_[cell]); // flame length starts at top of canopy - another assumption
//            z_o = simulation_->canopy_height_[cell];
            z_o = 0.4306;
//            std::cout << "Height: " << z_o << ' ';// << std::endl;
            velocity_ratio = (float) (simulation_->B * pow(simulation_->D_p / z_F, 0.5));// v_o/w_F
            r = (float)pow((simulation_->b_x+z_o/z_F)/simulation_->a_x, 0.5);

            t_0 = 0.7f; // Why did I pick this? For literally no reason. It's just 0.7.
            t_1 = (float) ( 1.f - pow((z_o / z_F), 0.5) +
                  velocity_ratio*log((1.f - velocity_ratio) / (pow((z_o / z_F), 0.5) - velocity_ratio)));
            t_2 = (float) (0.2f + simulation_->B*pow((simulation_->D_p / z_F), 0.5) *
                                     ( 1.0f + simulation_->B*pow((simulation_->D_p / z_F), 0.5) *
                                     log(1.f + 1.f/(1.f - pow((simulation_->D_p / z_F), 0.5))) ));
            t_3 = (float) (simulation_->a_x / (0.8*velocity_ratio) * ( log((1.f - 0.8*velocity_ratio)/(1.f - 0.8 * r * velocity_ratio))
                                                             - 0.8*velocity_ratio*(r - 1) -0.5f * pow(.8*velocity_ratio, 2) * pow(r-1, 2) ));
            t_t = t_0 + t_1 + t_2 + t_3;
//            std::cout << t_t << ' ' << t_0 << ' ' << t_1 << ' ' << t_2 << ' ' << t_3 << std::endl;

            // Find max height of particle (t_f = t_t)
//            max = (simulation_->b_x*z_F + z_F * simulation_->a_x * pow((3 * (t_t - t_0 - 1.2) )/ simulation_->a_x + 1.f, (2./3.)));
            max = simulation_->canopy_height_[cell] + 30.f;
            Ember launch(col, row, max, max,
                         atan2(simulation_->wind_t_[row][col].y, simulation_->wind_t_[row][col].x),
                         (pow((pow(simulation_->wind_t_[row][col].x,2) + pow(simulation_->wind_t_[row][col].y,2)),0.5)));

            // Add ember to tracked list of embers
            ember_list_.push_back(launch);
//            std::cout << row << ' ' << col << ' ' << t << ' ' << curspreadrate_[cell_sr] << std::endl;

            // mark map as 'launched'
            ember_map_[cell_sr] = true;
            test_map_[cell_sr] = t;
//            simulation_->ign_time_[cell] = 0;

         }
         // END SPOTTING CODE //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/
         float maxCrownRate = 3.34f * maxspreadrate_[cell_sr];
         float surfaceFuelConsumption =
               simulation_->I_o_[cell] * curspreadrate_[cell_sr] / I_b; // surfaceFuelConsumption is R0
         float crownCoefficient = (float) (
               -log(0.1) / (0.9 * (simulation_->RAC_[cell] - surfaceFuelConsumption))); // crownCoefficient is ac
         float crownFractionBurned =(float) (
               1.0 -
               exp(-crownCoefficient * (curspreadrate_[cell_sr] - surfaceFuelConsumption))); // crownFractionBurned is CFB
         crownFractionBurned = simulation_->Clamp(crownFractionBurned, 0.0, 1.0);
         float crownRate = curspreadrate_[cell_sr] + crownFractionBurned * (maxCrownRate - curspreadrate_[cell_sr]);
         if (crownRate >= simulation_->RAC_[cell]) { // passive -> active
//            maxspreadrate_[cell_sr] = max(crownRate, maxspreadrate_[cell_sr]);
            maxspreadrate_[cell_sr] = (crownRate > maxspreadrate_[cell_sr] ? crownRate : maxspreadrate_[cell_sr]);
//            cout << "Crown max updated" << endl;
         }
      }
   }

}


/*
Function: TestSpotting
Input: None
Purpose: If the fire has crowned, then this is called to test if spotting occurs and where
*/
bool SequentialSpread::TestSpotting(float t){
   double z_o, dX;
   z_o = 0.4306;

   for(std::vector<Ember>::iterator it = ember_list_.begin(); it != ember_list_.end();){
      int cell = simulation_->sim_dim_x_ * it->y_ + it->x_;
//      std::cout << it->x_ << ' ' << it->y_ << std::endl;
      // Check in bounds for sim:
      if(it->x_ >= sim_cols_ || it->y_ >= sim_rows_ || it->x_ < 0. || it->y_ < 0.){
         ember_list_.erase(it);
         continue;
      }

//      if(it->z_ <= it->z_o_){ // Impact
      if(it->z_ <= simulation_->canopy_height_[cell]){ // Impact
         // start fire ---------------------------------------------------------------------- CHANGE TO PROBABILITY LATER
//         simulation_->ign_time_[cell] = std::min(t, simulation_->ign_time_[cell]); // will set fire if not already set
         simulation_->ign_time_[cell] = t;
//         std::cout << "SUCCESS\n";
//         std::cout << it->x_ << ' ' << it->y_ << ' ' << cell <<  std::endl;
         // remove from list
         ember_list_.erase(it);
         continue;
      }
      // move ember
      dX = ((it->magnitude_ * log(it->z_ / z_o)) / log(simulation_->canopy_height_[cell] / z_o)); // rate of change
      dX = simulation_->time_step_ * dX; // magnitude of change -- will be different for all kernels :(
      it->x_ -=(int)(dX * cos(it->angle_)); // this will truncate the values to integers, but that's an acceptable error for now
      it->y_ -=(int)(dX * sin(it->angle_));
//      std::cout << (int)(dX * cos(it->angle_)) << ' ' << (int)(dX * sin(it->angle_)) << std::endl;

      // update height value
//      float sub = (float) (simulation_->v_o * (t / simulation_->tau - 0.5f * pow(t / simulation_->tau, 2)));
////      std::cout << sub << ' ' << t << ' ';
//      if(sub < 0) {
//         ember_list_.erase(it);
//         continue;
//      }
//      else
//         it->z_ = (float) (it->z_o_ - simulation_->v_o * ((t) / simulation_->tau - 0.5f * pow((t) / simulation_->tau, 2)));
         it->z_ -= 15.3f;
//      std::cout << it->z_ << ' ' << ((t) / simulation_->tau) - 0.5f * pow((t) / simulation_->tau, 2)
//                << " time: " << t << ' ' << simulation_->tau << std::endl;
      it++;
   }
}


