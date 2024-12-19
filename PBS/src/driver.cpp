/* Copyright (C) Jiaoyang Li
* Unauthorized copying of this file, via any medium is strictly prohibited
* Confidential
* Written by Jiaoyang Li <jiaoyanl@usc.edu>, May 2020
*/

/*driver.cpp
* Solve a MAPF instance on 2D grids.
*/
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include "PBS.h"


/* Main function */
int main(int argc, char** argv)
{
	namespace po = boost::program_options;
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")

		// params for the input instance and experiment settings
		("map,m", po::value<string>()->required(), "input file for map")
		//("agents,a", po::value<string>()->required(), "input file for agents")
		("output,o", po::value<string>(), "output file for statistics")
		("outputPaths", po::value<string>(), "output file for paths")
		("agentNum,k", po::value<int>()->default_value(0), "number of agents")
		("cutoffTime,t", po::value<double>()->default_value(7200), "cutoff time (seconds)")
		("screen,s", po::value<int>()->default_value(1), "screen option (0: none; 1: results; 2:all)")
		("stats", po::value<bool>()->default_value(false), "write to files some detailed statistics")

		("sipp", po::value<bool>()->default_value(1), "using SIPP as the low-level solver")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
		cout << desc << endl;
		return 1;
	}

	po::notify(vm);

	srand((int)time(0));

	///////////////////////////////////////////////////////////////////////////
	// load the instance
	std::ifstream i(vm["map"].as<std::string>());
        json map_json;
        i >> map_json;
	
	Instance instance(map_json, 150,1);

	// shuffle start and goal locations with a given random seed
	vector<int> start_locations = instance.get_start_locations();
	vector<int> goal_locations = instance.get_goal_locations();
	// std::cout << "start_locations: ";
    // for (const int& num : start_locations) { // Range-based for loop
    //     std::cout << num << " ";
    // }
    // std::cout << std::endl;

	// std::cout << "goal_locations: ";
    // for (const int& num : goal_locations) { // Range-based for loop
    //     std::cout << num << " ";
    // }
    // std::cout << std::endl;
	


	// Instance instance(vm["map"].as<string>(), vm["agents"].as<string>(),
	// 	vm["agentNum"].as<int>());

	srand(0);
	// int runs = 10;
    PBS pbs(instance, vm["sipp"].as<bool>(), vm["screen"].as<int>());
    // run
    double runtime = 0;
	pbs.solve(vm["cutoffTime"].as<double>());
	runtime += pbs.runtime;
	if (pbs.solution_cost == -2)
	{ 
		while (!pbs.solution_found)
		{
			instance.shuffle_start_goal(start_locations,goal_locations);
			pbs.clear();
			pbs.solve(vm["cutoffTime"].as<double>()-runtime);
			runtime += pbs.runtime;	
			if (pbs.solution_cost == -1 || pbs.solution_cost >= 0 || pbs.solution_cost == -3)
			{
				break;
			} 
			cout << "Failed to find solutions with runtime " << pbs.runtime << endl;	
		}
	}
	// for (int i = 0; i < runs; i++)
	// {
	// 	instance.shuffle_start_goal(start_locations,goal_locations);
	// 	pbs.clear();
	// 	pbs.solve(vm["cutoffTime"].as<double>()-runtime);
	// 	runtime += pbs.runtime;
	// 	if (pbs.solution_found)
	// 		break;
	// 	cout << "Failed to find solutions in Run " << i << endl;
	
	// }
    
	pbs.runtime = runtime;

	cout << "runtime: " << pbs.runtime << endl;

    if (vm.count("output"))
        pbs.saveResults(vm["output"].as<string>());
	cout << "Solution Found: " << pbs.solution_found << endl;
    if (pbs.solution_found && vm.count("outputPaths"))
        pbs.savePaths(vm["outputPaths"].as<string>());

    /*size_t pos = vm["output"].as<string>().rfind('.');      // position of the file extension
    string output_name = vm["output"].as<string>().substr(0, pos);     // get the name without extension
    cbs.saveCT(output_name); // for debug*/

    pbs.clearSearchEngines();

	return 0;

}