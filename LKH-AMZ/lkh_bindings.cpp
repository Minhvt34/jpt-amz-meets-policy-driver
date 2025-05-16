#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <iostream>

// Forward declaration for LKH API
extern "C" {
    #include "SRC/INCLUDE/LKH.h"
    // Expected LKH globals and functions:
    // extern char* ProblemFileName;
    // extern char* ParameterFileName;
    // extern void ReadParameters();
    // extern void ReadProblem();
    // extern long long FindTour();
    // extern void FreeStructures(); // (If it exists and is needed)
}

// Using pybind11 namespace directly for arg policies
// namespace py = pybind11; // This can sometimes confuse linters in module scope

// Define the core logic as a separate C++ function
long long solve_lkh_tsp_impl(const std::string& problem_file_path, const std::string& parameter_file_path) {
    printf("[LKH_BINDING_DEBUG] Entering solve_lkh_tsp_impl.\n");
    fflush(stdout);

    if (problem_file_path.empty()) {
        throw std::runtime_error("Problem file path cannot be empty.");
    }
    if (parameter_file_path.empty()) {
        throw std::runtime_error("Parameter file path cannot be empty.");
    }

    printf("[LKH_BINDING_DEBUG] Problem file: %s\n", problem_file_path.c_str());
    printf("[LKH_BINDING_DEBUG] Parameter file: %s\n", parameter_file_path.c_str());
    fflush(stdout);

    FILE* problem_file_check_ptr = fopen(problem_file_path.c_str(), "r");
    if (!problem_file_check_ptr) {
        throw std::runtime_error("Could not open problem file for verification: " + problem_file_path);
    }
    fclose(problem_file_check_ptr);
    printf("[LKH_BINDING_DEBUG] Problem file verified by fopen/fclose.\n");
    fflush(stdout);

    FILE* param_file_check_ptr = fopen(parameter_file_path.c_str(), "r");
    if (!param_file_check_ptr) {
        throw std::runtime_error("Could not open parameter file for verification: " + parameter_file_path);
    }
    fclose(param_file_check_ptr);
    printf("[LKH_BINDING_DEBUG] Parameter file verified by fopen/fclose.\n");
    fflush(stdout);

    // Initialize random seed - LKH's main usually does this.
    // Seed is a global variable from LKH.h, usually set by ReadParameters or command line.
    // For now, let's try a fixed seed before ReadParameters in case it matters.
    // If Seed is properly set by ReadParameters later, this might be redundant or overridden.
    SRandom(1); // Using a fixed seed, e.g., 1
    printf("[LKH_BINDING_DEBUG] Called SRandom(1).\n");
    fflush(stdout);

    ParameterFileName = const_cast<char*>(parameter_file_path.c_str());
    ProblemFileName = const_cast<char*>(problem_file_path.c_str()); 
    printf("[LKH_BINDING_DEBUG] Global LKH file names set.\n");
    fflush(stdout);

    printf("[LKH_BINDING_DEBUG] Calling ReadParameters()...\n");
    fflush(stdout);
    ReadParameters();    
    printf("[LKH_BINDING_DEBUG] ReadParameters() completed.\n");
    fflush(stdout);

    printf("[LKH_BINDING_DEBUG] Calling ReadProblem()...\n");
    fflush(stdout);
    ReadProblem();    
    printf("[LKH_BINDING_DEBUG] ReadProblem() completed.\n");
    fflush(stdout);
    
    printf("[LKH_BINDING_DEBUG] Allocating structures...\n");
    fflush(stdout);
    AllocateStructures();
    printf("[LKH_BINDING_DEBUG] Creating candidate set...\n");
    fflush(stdout);
    CreateCandidateSet();
    
    printf("[LKH_BINDING_DEBUG] Calling FindTour()...\n");
    fflush(stdout);
    long long tour_cost = FindTour();
    printf("[LKH_BINDING_DEBUG] FindTour() completed. Cost: %lld\n", tour_cost);
    fflush(stdout);
    
    // Example for calling FreeStructures, if it exists and is needed.
    // This function might not be present in all LKH versions or might have a different name.
    // Ensure LKH.h actually declares FreeStructures() for this to be valid.
    /*
    if (FreeStructures) { 
        FreeStructures(); 
    }
    */

    return tour_cost;
}

// PYBIND11_MODULE definition
PYBIND11_MODULE(lkh_solver, mod) {
    pybind11::module_& m = mod; // Explicitly use pybind11::module_ and assign to m
    m.doc() = "Python bindings for the LKH Traveling Salesman Problem solver";

    m.def("solve_tsp", 
          &solve_lkh_tsp_impl, 
          "Solves a TSP problem using a problem file and a parameter file, returning the tour cost.",
          pybind11::arg("problem_file_path"), 
          pybind11::arg("parameter_file_path")
    );
} 