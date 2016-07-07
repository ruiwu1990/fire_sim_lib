#include <iostream>
#include <fstream>
#include "FireSim.h"
//#include <sys/time.h>
//#include "kernel_MT.h"
//#include "kernel_IMT.h"
#include "BD.h"
#include "MT.h"
#include "IMT.h"
#include "SEQ.h"
#include <sys/time.h>

bool TIMING_LOOPS = 1;

bool PARALLEL = 1;

int main(int argc, char *argv[]){
//   int step_size = 350;
//   int s = 1024;
   int step_size = 600;
   double t_upSpread;
   int s = 512;
   std::ofstream fout;
   std::string filename;
   filename += "/cse/home/andyg/Desktop/firesim/out/timeout.txt";
   fout.open(filename.c_str());
   for(int i = 0; i < TIMING_LOOPS; i ++)
   {
      if(PARALLEL)
      {
         printf("---------- Running Parallel Simulation ----------\n");
         BD parallel_sim(906,642,
                         "/cse/home/andyg/Desktop/firesim/data/default.fmd",
                         "/cse/home/andyg/Desktop/firesim/data/kyle.fms");
         parallel_sim.Init("/cse/home/andyg/Desktop/firesim/data/fixed.fuel", //--> argv[1]
                           "/cse/home/andyg/Desktop/firesim/data/fixed2.tif",
                           "/cse/home/andyg/Desktop/firesim/data/canopy_ht.asc",
                           "/cse/home/andyg/Desktop/firesim/data/crown_base_ht.asc",
                           "/cse/home/andyg/Desktop/firesim/data/crown_bulk_density.asc",
                           0, 0);

         // check for rui file
         //newData = "/cse/home/andyg/Desktop/simulator/data/" + argv[1];
         // path = argv[2]; // file path 'pwd'
         // newData = argv[2] + "/simulator/data/" + argv[3]; // argv[2] is file name .asc, .csv ...
         // newData = "/cse/home/andyg/Desktop/" + "simulator/data/" + fire_info.csv

         // if there is a file parse it

         // Loop through and start fires
        std::ifstream finn("/cse/home/andyg/Desktop/firesim/data/fire_info.csv");
        std::ofstream foutt("/cse/home/andyg/Desktop/firesim/data/temp.txt");

        if(!finn)
        {
          std::cerr<<"Failed to open file !";
          return 1;
        }

        char temp[255];
        double left_top_lat, left_top_long, right_bottom_lat, right_bottom_long;
        int numrows, numcols, lmaxval, notsetfire;

        finn.getline(temp, 255, ':');
        finn >> left_top_lat;
        std::cout << std::endl << temp << " " << left_top_lat << std::endl;

        finn.getline(temp, 255, ':');
        finn >> left_top_long;
        std::cout << temp << " " << left_top_long << std::endl;

        finn.getline(temp, 255, ':');
        finn >> right_bottom_lat;
        std::cout << temp << " " << right_bottom_lat << std::endl;

        finn.getline(temp, 255, ':');
        finn >> right_bottom_long;
        std::cout << temp << " " << right_bottom_long << std::endl;

        finn.getline(temp, 255, ':');
        finn >> numrows;
        std::cout << temp << " " << numrows << std::endl;

        finn.getline(temp, 255, ':');
        finn >> numcols;
        std::cout << temp << " " << numcols << std::endl;

        finn.getline(temp, 255, ':');
        finn >> lmaxval;
        std::cout << temp << " " << lmaxval << std::endl;

        finn.getline(temp, 255, ':');
        finn >> notsetfire;
        std::cout << temp << " " << notsetfire << std::endl;

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
          std::cout << "cols: " << c << std::endl;
          c = 0;
          r++;
          std::cout << "rows: " << r << std::endl;
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

         parallel_sim.WriteToFile("/cse/home/andyg/Desktop/firesim/out/final_tests.csv"); // add meta data for rui --> argv[5] maybe?
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