#include <pybind11/pybind11.h>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include "PBS.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace py = pybind11;

/* Main function */
std::string run(const py::kwargs& kwargs)
{
	namespace po = boost::program_options;
    json result;
	std::vector<double> init_tile_usage;
	// Declare the supported options.
	// po::options_description desc("Allowed options");
	// desc.add_options()
	// 	("help", "produce help message")

	// 	// params for the input instance and experiment settings
	// 	("map,m", po::value<string>()->required(), "input file for map")
	// 	("agents,a", po::value<string>()->required(), "input file for agents")
	// 	("output,o", po::value<string>(), "output file for statistics")
	// 	("outputPaths", po::value<string>(), "output file for paths")
	// 	("agentNum,k", po::value<int>()->default_value(0), "number of agents")
	// 	("cutoffTime,t", po::value<double>()->default_value(7200), "cutoff time (seconds)")
	// 	("screen,s", po::value<int>()->default_value(1), "screen option (0: none; 1: results; 2:all)")
	// 	("stats", po::value<bool>()->default_value(false), "write to files some detailed statistics")

	// 	("sipp", po::value<bool>()->default_value(1), "using SIPP as the low-level solver")
	// 	;
	result["tile_usage"] = init_tile_usage;
    result["tile_usage_mean"] = 0.0;
    result["tile_usage_std"] = 0.0;


    result["finish"] = false;
	// po::variables_map vm;
	// po::store(po::parse_command_line(argc, argv, desc), vm);

	// if (vm.count("help")) {
	// 	cout << desc << endl;
	// 	return 1;
	// }

	// po::notify(vm);

	srand((int)time(0));

	///////////////////////////////////////////////////////////////////////////
	// load the instance
	// Instance instance(kwargs["map"].cast<string>(), kwargs["agents"].cast<string>(),
	// 	kwargs["agentNum"].cast<int>());

	json G_json = json::parse(kwargs["map"].cast<string>());
	Instance instance(G_json, kwargs["agentNum"].cast<int>());

	srand(0);
    PBS pbs(instance, kwargs["sipp"].cast<bool>(), kwargs["screen"].cast<int>());
    // run
    double runtime = 0;
    pbs.solve(kwargs["cutoffTime"].cast<double>());
    if (kwargs.contains("output"))
        pbs.saveResults(kwargs["output"].cast<string>());
    if (pbs.solution_found && kwargs.contains("outputPaths"))
        pbs.savePaths(kwargs["outputPaths"].cast<string>());
    /*size_t pos = vm["output"].as<string>().rfind('.');      // position of the file extension
    string output_name = vm["output"].as<string>().substr(0, pos);     // get the name without extension
    cbs.saveCT(output_name); // for debug*/
	runtime += pbs.runtime;

	result =  pbs.cal_tile_usage(instance, result);
	result["suboptimality"] = pbs.cal_suboptimality(instance,pbs.solution_cost);
	result["solution_cost"] = pbs.solution_cost;
    result["runtime"] = runtime;
    result["finish"] = true;
	result["solution_found"] = pbs.solution_found;
	result["start_locations"] = instance.get_start_locations();
	result["goal_locations"] = instance.get_goal_locations();

    pbs.clearSearchEngines();

	return result.dump(4);

}

string playground(){
	std::string json_string = R"(
	{
		"pi": 3.141,
		"happy": true
	}
	)";
	json ex1 = json::parse(json_string);

	cout << ex1["pi"] << endl;

	return ex1.dump();
}


PYBIND11_MODULE(warehouse_sim_PBS, m) {
	// optional module docstring
    // m.doc() = ;

    m.def("playground", &playground, "Playground function to test everything");
    // m.def("add", &add, py::arg("i")=0, py::arg("j")=1);
    m.def("run", &run, "Function to run warehouse simulation");
}