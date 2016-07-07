#include <iostream>
#include <fstream>
#include "FireSim.h"
#include <sstream>
#include <string>
#include <cstring>
//#include <sys/time.h>
//#include "kernel_MT.h"
//#include "kernel_IMT.h"
#include "BD.h"
#include "MT.h"
#include "IMT.h"
#include "SEQ.h"
#include <sys/time.h>

/*
./simulator /cse/home/andyg/Desktop/simulator/data/fixed.fuel /cse/home/andyg/Desktop/simulator/data/fire_info.csv /cse/home/andyg/Desktop/simulator/out/final_tests.csv
*/

bool TIMING_LOOPS = 1;

bool PARALLEL = 1;

std::string double_to_string(double value)
{
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

std::string int_to_string(int value)
{
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

int main(int argc, char *argv[]){
//   int step_size = 350;
//   int s = 1024;
   int step_size = 600;
   double t_upSpread;
   int s = 512;
   std::ofstream fout;
   std::string filename;
   filename += "/cse/home/andyg/Desktop/simulator/out/timeout.txt";
   fout.open(filename.c_str());
   for(int i = 0; i < TIMING_LOOPS; i ++)
   {
      if(PARALLEL)
      {
         printf("---------- Running Parallel Simulation ----------\n");
         BD parallel_sim(906,642,
                         "/cse/home/andyg/Desktop/simulator/data/default.fmd", // /cse/home/andyg/Desktop/ + argv[4] this does not work
                                                                               // need to get path for files
                         "/cse/home/andyg/Desktop/simulator/data/kyle.fms");
         parallel_sim.Init(argv[1], //--> argv[1] --> "/cse/home/andyg/Desktop/simulator/data/fixed.fuel"
                           "/cse/home/andyg/Desktop/simulator/data/fixed2.tif",
                           "/cse/home/andyg/Desktop/simulator/data/canopy_ht.asc",
                           "/cse/home/andyg/Desktop/simulator/data/crown_base_ht.asc",
                           "/cse/home/andyg/Desktop/simulator/data/crown_bulk_density.asc",
                           0, 0);

         // check for rui file
         //newData = "/cse/home/andyg/Desktop/simulator/data/" + argv[1];
         // path = argv[2]; // file path 'pwd'
         // newData = argv[2] + "/simulator/data/" + argv[3]; // argv[2] is file name .asc, .csv ...
         // newData = "/cse/home/andyg/Desktop/" + "simulator/data/" + fire_info.csv

         // if there is a file parse it

         // Loop through and start fires
        std::ifstream finn(argv[2]); // "/cse/home/andyg/Desktop/simulator/data/fire_info.csv"
        std::ofstream foutt("/cse/home/andyg/Desktop/simulator/data/temp.txt");

        if(!finn)
        {
          std::cerr<<"Failed to open file !";
          return 1;
        }

        char temp[255];
        double left_top_lat, left_top_long, right_bottom_lat, right_bottom_long;
        int numrows, numcols, lmaxval, notsetfire;
        std::string* metaData = new std::string[8];
        std::string spaceColon = ": ";

        finn.getline(temp, 255, ':');
        finn >> left_top_lat;
        std::cout << std::endl << temp << " " << left_top_lat << std::endl;
        metaData[0] = temp + spaceColon + double_to_string(left_top_lat);
        std::cout << metaData[0] << std::endl;

        finn.getline(temp, 255, ':');
        finn >> left_top_long;
        std::cout << temp << " " << left_top_long << std::endl;
        metaData[1] = temp + spaceColon + double_to_string(left_top_long);
        std::cout << metaData[1] << std::endl;

        finn.getline(temp, 255, ':');
        finn >> right_bottom_lat;
        std::cout << temp << " " << right_bottom_lat << std::endl;
        metaData[2] = temp + spaceColon + double_to_string(right_bottom_lat);
        std::cout << metaData[2] << std::endl;

        finn.getline(temp, 255, ':');
        finn >> right_bottom_long;
        std::cout << temp << " " << right_bottom_long << std::endl;
        metaData[3] = temp + spaceColon + double_to_string(right_bottom_long);
        std::cout << metaData[3] << std::endl;

        finn.getline(temp, 255, ':');
        finn >> numrows;
        std::cout << temp << " " << numrows << std::endl;
        metaData[4] = temp + spaceColon + int_to_string(numrows);
        std::cout << metaData[4] << std::endl;

        finn.getline(temp, 255, ':');
        finn >> numcols;
        std::cout << temp << " " << numcols << std::endl;
        metaData[5] = temp + spaceColon + int_to_string(numcols);
        std::cout << metaData[5] << std::endl;

        finn.getline(temp, 255, ':');
        finn >> lmaxval;
        std::cout << temp << " " << lmaxval << std::endl;
        metaData[6] = temp + spaceColon + int_to_string(lmaxval);
        std::cout << metaData[6] << std::endl;

        finn.getline(temp, 255, ':');
        finn >> notsetfire;
        std::cout << temp << " " << notsetfire << std::endl;
        metaData[7] = temp + spaceColon + int_to_string(notsetfire);
        std::cout << metaData[7] << std::endl;

        int count = 0, r = 0, c = 0;
        while(r < numrows)
        {
          int value;
          char dummy;
          while(c < numcols-1)
          {      
            finn >> value >> dummy;
            if(value == 1)
            {
             parallel_sim.UpdateCell(r,c,0);
             count++;
            }
            c++;
            foutt << value << ",";
          }
          finn >> value;
          if(value == 1)
            {
             parallel_sim.UpdateCell(r,c,0);
             count++;
            }
          //std::cout << "cols: " << c << std::endl;
          c = 0;
          r++;
          //std::cout << "rows: " << r << std::endl;
          foutt << value << std::endl;
        }
         std::cout << count << std::endl;
         finn.close();

         struct timeval start, fin;
         gettimeofday(&start, NULL);

         parallel_sim.CopyToDevice();

         for(int i = 0; i < 1; i++)
         {
            parallel_sim.RunKernel(step_size, s, s, true);
//           std::cout << " Count: " << i << std::endl;
         }

         parallel_sim.CopyFromDevice();

         gettimeofday(&fin, NULL);
         t_upSpread = fin.tv_usec + fin.tv_sec * 1000000.0;
         t_upSpread -= start.tv_usec + start.tv_sec * 1000000.0;
         t_upSpread /= 1000000.0;
         std::cout << "Processing Simulation took " << t_upSpread << " seconds" << std::endl;
//       std::cout << "Iterations: " << count << std::endl;

         parallel_sim.WriteToFile(argv[3], metaData); // "/cse/home/andyg/Desktop/simulator/out/final_tests.csv"
         delete[] metaData;
      }
      else
      {
         printf("---------- Running Sequential Simulation ----------\n");
         SequentialSpread sequential_sim(128, 128);
         sequential_sim.Init();
         sequential_sim.CalcMaxSpreadRates();
         struct timeval start, fin;
         gettimeofday(&start, NULL);

//      sequential_sim.RunSimulationIMT(step_size);
         sequential_sim.RunSimulationBD(step_size);
//      sequential_sim.RunSimulationMT(step_size);

         gettimeofday(&fin, NULL);
         double t_upSpread = fin.tv_usec + fin.tv_sec * 1000000.0;
         t_upSpread -= start.tv_usec + start.tv_sec * 1000000.0;
         t_upSpread /= 1000000.0;
         std::cout << "Processing Simulation took " << t_upSpread << " seconds" << std::endl;
//         std::printf("Processing Simulation Took: %f seconds", t_upSpread);
         fout << t_upSpread << ',';
         fout.close();
         sequential_sim.WriteToFile();
      }
   }

   std::cout << "Simulation Complete" << std::endl;

   return 0;
}
/*int main(int argc, char *argv[])
{
   char yesno;
   int step_size = 600;
   double t_upSpread;
   int s = 512;
   std::ofstream fout;
   std::ifstream fin;
   std::string filename, newData;
   filename += "/cse/home/andyg/Desktop/simulator/out/timeout.txt";
   fout.open(filename.c_str());

   while(RUN_SIM == 1)
   {
      printf("---------- Running Parallel Simulation ----------\n");
      BD parallel_sim(906,642,
                      "/cse/home/andyg/Desktop/simulator/data/default.fmd",
                      "/cse/home/andyg/Desktop/simulator/data/kyle.fms");
      parallel_sim.Init("/cse/home/andyg/Desktop/simulator/data/fixed.fuel",
                        "/cse/home/andyg/Desktop/simulator/data/fixed2.tif",
                        "/cse/home/andyg/Desktop/simulator/data/canopy_ht.asc",
                        "/cse/home/andyg/Desktop/simulator/data/crown_base_ht.asc",
                        "/cse/home/andyg/Desktop/simulator/data/crown_bulk_density.asc",
                        0, 0);

      // check for rui file
      newData = "/cse/home/andyg/Desktop/simulator/data/" + newData;

      // if there is a file parse it

      // Loop through and start fires
      parallel_sim.UpdateCell(320,780,0);

      struct timeval start, fin;
      gettimeofday(&start, NULL);

      parallel_sim.CopyToDevice();

      for(int i = 0; i < 1; i++)
      {
         parallel_sim.RunKernel(step_size, s, s, true);
      }

      parallel_sim.CopyFromDevice();

      gettimeofday(&fin, NULL);
      t_upSpread = fin.tv_usec + fin.tv_sec * 1000000.0;
      t_upSpread -= start.tv_usec + start.tv_sec * 1000000.0;
      t_upSpread /= 1000000.0;
      std::cout << "Processing Simulation took " << t_upSpread << " seconds" << std::endl;

      parallel_sim.WriteToFile("/cse/home/andyg/Desktop/simulator/out/final_tests.csv"); // add meta data for rui

      std::cout << std::endl << "Simulation Complete, would you like to restart with new data? (y/n)" << std::endl;
      cin >> yesno;
      if(yesno == 'y')
      {
         std::cout << "Please enter filename for new data:" << std::endl;
         std::cin >> newData;
         RUN_SIM = 1;
      }
      else
      {
         RUN_SIM = 0;
      }

   }

   std::cout << "Simulation Shutdown" << std::endl;
   return 0;
}
*/
/*
int main(int argc, char *argv[]){
//   int step_size = 350;
//   int s = 1024;
   int step_size = 600;
   double t_upSpread;
   int s = 512;
   std::ofstream fout;
   std::string filename;
   filename += "/cse/home/andyg/Desktop/simulator/out/timeout.txt";
   fout.open(filename.c_str());
   for(int i = 0; i < TIMING_LOOPS; i ++)
   {
      if(PARALLEL)
      {
         printf("---------- Running Parallel Simulation ----------\n");
         BD parallel_sim(906,642,
                         "/cse/home/andyg/Desktop/simulator/data/default.fmd",
                         "/cse/home/andyg/Desktop/simulator/data/kyle.fms");
         parallel_sim.Init("/cse/home/andyg/Desktop/simulator/data/fixed.fuel",
                           "/cse/home/andyg/Desktop/simulator/data/fixed2.tif",
                           "/cse/home/andyg/Desktop/simulator/data/canopy_ht.asc",
                           "/cse/home/andyg/Desktop/simulator/data/crown_base_ht.asc",
                           "/cse/home/andyg/Desktop/simulator/data/crown_bulk_density.asc",
                           0, 0);

         // check for rui file

         // if there is a file parse it

         // Loop through and start fires
         parallel_sim.UpdateCell(320,780,0);

         struct timeval start, fin;
         gettimeofday(&start, NULL);

         parallel_sim.CopyToDevice();

         for(int i = 0; i < 1; i++)
         {
            parallel_sim.RunKernel(step_size, s, s, true);
//            std::cout << " Count: " << i << std::endl;
         }

         parallel_sim.CopyFromDevice();

         gettimeofday(&fin, NULL);
         t_upSpread = fin.tv_usec + fin.tv_sec * 1000000.0;
         t_upSpread -= start.tv_usec + start.tv_sec * 1000000.0;
         t_upSpread /= 1000000.0;
         std::cout << "Processing Simulation took " << t_upSpread << " seconds" << std::endl;
//       std::cout << "Iterations: " << count << std::endl;

         parallel_sim.WriteToFile("/cse/home/andyg/Desktop/simulator/out/final_tests.csv"); // add meta data for rui
      }
      else
      {
         printf("---------- Running Sequential Simulation ----------\n");
         SequentialSpread sequential_sim(128, 128);
         sequential_sim.Init();
         sequential_sim.CalcMaxSpreadRates();
         struct timeval start, fin;
         gettimeofday(&start, NULL);

//      sequential_sim.RunSimulationIMT(step_size);
         sequential_sim.RunSimulationBD(step_size);
//      sequential_sim.RunSimulationMT(step_size);

         gettimeofday(&fin, NULL);
         double t_upSpread = fin.tv_usec + fin.tv_sec * 1000000.0;
         t_upSpread -= start.tv_usec + start.tv_sec * 1000000.0;
         t_upSpread /= 1000000.0;
         std::cout << "Processing Simulation took " << t_upSpread << " seconds" << std::endl;
//         std::printf("Processing Simulation Took: %f seconds", t_upSpread);
         fout << t_upSpread << ',';
         fout.close();
         sequential_sim.WriteToFile();
      }
   }

   std::cout << "Simulation Complete" << std::endl;

   return 0;
}
*/