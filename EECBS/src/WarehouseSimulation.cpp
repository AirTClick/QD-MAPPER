// Main API for python to run simulation

#include <pybind11/pybind11.h>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include "CBS.h"
#include "ECBS.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace py = pybind11;

std::string run(const py::kwargs& kwargs) {
	// if (kwargs["test"].cast<bool>())
	// {
	// 	// For testing
	// 	json result = {
    //         {"throughput", 10},
	// 		{"tile_usage", {0.0, 1.0, 2.0}},
	// 		{"num_wait", {3.0, 4.0, 5.0}},
	// 		{"finished_task_len", {6.0, 7.0, 8.0}},
	// 		{"tile_usage_mean", 1.0},
	// 		{"tile_usage_std", 0.67},
	// 		{"num_wait_mean", 4.0},
	// 		{"num_wait_std", 0.67},
	// 		{"finished_len_mean", 7.0},
	// 		{"finished_len_std", 0.67}
	// 	};

	// 	return result.dump(4);
	// }

    namespace po = boost::program_options;
    clock_t start_time = clock();
	json result;
    std::vector<double> init_tile_usage;

	result["finish"] = false; // status code to false by default
    result["solution_cost"] = -2.0;
    result["runtime"] = 0;
    result["solution_found"] = false;
    result["suboptimality"] = 1.0;
    
    result["tile_usage"] = init_tile_usage;
    result["tile_usage_mean"] = 0.0;
    result["tile_usage_std"] = 0.0;


    high_level_solver_type s;
	if (kwargs["highLevelSolver"].cast<string>() == "A*")
		s = high_level_solver_type::ASTAR;
	else if (kwargs["highLevelSolver"].cast<string>() == "A*eps")
		s = high_level_solver_type::ASTAREPS;
	else if (kwargs["highLevelSolver"].cast<string>() == "EES")
		s = high_level_solver_type::EES;
	else if (kwargs["highLevelSolver"].cast<string>() == "NEW")
		s = high_level_solver_type::NEW;
	else
	{
		cout << "WRONG high level solver!" << endl;
		exit(-1);
	}

	if (s == high_level_solver_type::ASTAR && kwargs["suboptimality"].cast<double>() > 1)
	{
		cerr << "A* cannot perform suboptimal search!" << endl;
		exit(-1);
	}

	heuristics_type h;
	if (kwargs["heuristics"].cast<string>() == "Zero")
		h = heuristics_type::ZERO;
	else if (kwargs["heuristics"].cast<string>() == "CG")
		h = heuristics_type::CG;
	else if (kwargs["heuristics"].cast<string>() == "DG")
		h = heuristics_type::DG;
	else if (kwargs["heuristics"].cast<string>() == "WDG")
		h = heuristics_type::WDG;
	else
	{
		cout << "WRONG heuristics strategy!" << endl;
		exit(-1);
	}

    if ((h == heuristics_type::CG || h == heuristics_type::DG) && kwargs["lowLevelSolver"].cast<bool>())
    {
        cerr << "CG or DG heuristics do not work with low level of suboptimal search!" << endl;
        exit(-1);
    }

	heuristics_type h_hat; // inadmissible heuristics
	if (s == high_level_solver_type::ASTAR ||
	    s == high_level_solver_type::ASTAREPS ||
	    kwargs["inadmissibleH"].cast<string>() == "Zero")
		h_hat = heuristics_type::ZERO;
	else if (kwargs["inadmissibleH"].cast<string>() == "Global")
		h_hat = heuristics_type::GLOBAL;
	else if (kwargs["inadmissibleH"].cast<string>() == "Path")
		h_hat = heuristics_type::PATH;
	else if (kwargs["inadmissibleH"].cast<string>() == "Local")
		h_hat = heuristics_type::LOCAL;
	else if (kwargs["inadmissibleH"].cast<string>() == "Conflict")
		h_hat = heuristics_type::CONFLICT;
	else
	{
		cout << "WRONG inadmissible heuristics strategy!" << endl;
		exit(-1);
	}

	conflict_selection conflict = conflict_selection::EARLIEST;
	node_selection n = node_selection::NODE_CONFLICTPAIRS;


	srand((int)time(0));

	///////////////////////////////////////////////////////////////////////////
	// load the instance
	// Instance instance(kwargs["map"].cast<string>(), kwargs["agents"].cast<string>(),
	// 	kwargs["agentNum"].cast<int>());

    json G_json = json::parse(kwargs["map"].cast<string>());
    // std::ifstream i(kwargs["map"].cast<std::string>());
    //     json map_json;
    //     i >> map_json;
	
	Instance instance(G_json, kwargs["agentNum"].cast<int>(),kwargs["eval_id"].cast<int>());

	srand(0);
	int runs = 1 + kwargs["restart"].cast<int>();
	//////////////////////////////////////////////////////////////////////
    // initialize the solver
	if (kwargs["lowLevelSolver"].cast<bool>())
    {
        ECBS ecbs(instance, kwargs["sipp"].cast<bool>(), kwargs["screen"].cast<int>());
        ecbs.setPrioritizeConflicts(kwargs["prioritizingConflicts"].cast<bool>());
        ecbs.setDisjointSplitting(kwargs["disjointSplitting"].cast<bool>());
        ecbs.setBypass(kwargs["bypass"].cast<bool>());
        ecbs.setRectangleReasoning(kwargs["rectangleReasoning"].cast<bool>());
        ecbs.setCorridorReasoning(kwargs["corridorReasoning"].cast<bool>());
        ecbs.setHeuristicType(h, h_hat);
        ecbs.setTargetReasoning(kwargs["targetReasoning"].cast<bool>());
        ecbs.setMutexReasoning(false);
        ecbs.setConflictSelectionRule(conflict);
        ecbs.setNodeSelectionRule(n);
        ecbs.setSavingStats(kwargs["stats"].cast<bool>());
        ecbs.setHighLevelSolver(s, kwargs["suboptimality"].cast<double>());
        //////////////////////////////////////////////////////////////////////
        // run
        double runtime = 0;
        int lowerbound = 0;
        for (int i = 0; i < runs; i++)
        {
            ecbs.clear();
            ecbs.solve(kwargs["cutoffTime"].cast<double>() / runs, lowerbound);
            runtime += ecbs.runtime;
            if (ecbs.solution_found)
                break;
            lowerbound = ecbs.getLowerBound();
            ecbs.randomRoot = true;
            cout << "Failed to find solutions in Run " << i << endl;
        }
        ecbs.runtime = runtime;
        //store tile_usage, mean, std inside result
        result =  ecbs.cal_tile_usage(instance, result);
        result["suboptimality"] = ecbs.cal_suboptimality(instance,ecbs.solution_cost);
        
        // if (runtime >= 5.0)
        // {
        //     result["suboptimality"] = -1.0;
        // }

        // cout << "For suboptimality: " << result["suboptimality"] << endl; 
        // cout << "Suboptimality: " << result["suboptimality"] << endl;

        if (kwargs.contains("output"))
            ecbs.saveResults(kwargs["output"].cast<string>());
        if (ecbs.solution_found && kwargs.contains("outputPaths"))
            ecbs.savePaths(kwargs["outputPaths"].cast<string>());
        /*size_t pos = kwargs["output"].as<string>().rfind('.');      // position of the file extension
        string output_name = kwargs["output"].as<string>().substr(0, pos);     // get the name without extension
        cbs.saveCT(output_name); // for debug*/
        if (kwargs["stats"].cast<bool>())
            ecbs.saveStats(kwargs["output"].cast<string>());
        
        result["runtime"] = ecbs.runtime;
        result["finish"] = true;
        result["solution_cost"] = ecbs.solution_cost;
        result["solution_found"] = ecbs.solution_found;
        result["start_locations"] = instance.get_start_locations();
        result["goal_locations"] = instance.get_goal_locations();


        // cout << "For finish: " << result["finish"] << endl;

        //test for tile_usage 
        // std::cout << "Tile_usage Vector: ";
		// for (const auto& num : result["tile_usage"]) {
		// 	std::cout << num << " ";
		// }
        
        ecbs.clearSearchEngines();
        
        cout << "Clean Done " <<  endl; 
    
    }
    else
    {
        CBS cbs(instance, kwargs["sipp"].cast<bool>(), kwargs["screen"].cast<int>());
        cbs.setPrioritizeConflicts(kwargs["prioritizingConflicts"].cast<bool>());
        cbs.setDisjointSplitting(kwargs["disjointSplitting"].cast<bool>());
        cbs.setBypass(kwargs["bypass"].cast<bool>());
        cbs.setRectangleReasoning(kwargs["rectangleReasoning"].cast<bool>());
        cbs.setCorridorReasoning(kwargs["corridorReasoning"].cast<bool>());
        cbs.setHeuristicType(h, h_hat);
        cbs.setTargetReasoning(kwargs["targetReasoning"].cast<bool>());
        cbs.setMutexReasoning(false);
        cbs.setConflictSelectionRule(conflict);
        cbs.setNodeSelectionRule(n);
        cbs.setSavingStats(kwargs["stats"].cast<bool>());
        cbs.setHighLevelSolver(s, kwargs["suboptimality"].cast<double>());
        //////////////////////////////////////////////////////////////////////
        // run
        double runtime = 0;
        int lowerbound = 0;
        for (int i = 0; i < runs; i++)
        {
            cbs.clear();
            cbs.solve(kwargs["cutoffTime"].cast<double>() / runs, lowerbound);
            runtime += cbs.runtime;
            if (cbs.solution_found)
                break;
            lowerbound = cbs.getLowerBound();
            cbs.randomRoot = true;
            cout << "Failed to find solutions in Run " << i << endl;
        }
        cbs.runtime = runtime;
        //store tile_usage, mean, std inside result
        result = cbs.cal_tile_usage(instance, result);

        if (kwargs.contains("output"))
            cbs.saveResults(kwargs["output"].cast<string>());
        if (cbs.solution_found && kwargs.contains("outputPaths"))
            cbs.savePaths(kwargs["outputPaths"].cast<string>());
        if (kwargs["stats"].cast<bool>())
            cbs.saveStats(kwargs["output"].cast<string>());
        
        result["runtime"] = cbs.runtime;
        result["solution_cost"] = cbs.solution_cost;
        result["finish"] = true;
        result["solution_found"] = cbs.solution_found;
        result["start_locations"] = instance.get_start_locations();
        result["goal_locations"] = instance.get_goal_locations();

        cout << "For finish: " << result["finish"] << endl;

        //test for tile_usage 
        // std::cout << "Tile_usage Vector: ";
		// for (const auto& num : result["tile_usage"]) {
		// 	std::cout << num << " ";
		// }

        cbs.clearSearchEngines();

        cout << "Clean Done " <<  endl; 
    }

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


PYBIND11_MODULE(warehouse_sim_EECBS, m) {
	// optional module docstring
    // m.doc() = ;

    m.def("playground", &playground, "Playground function to test everything");
    // m.def("add", &add, py::arg("i")=0, py::arg("j")=1);
    m.def("run", &run, "Function to run warehouse simulation");
}