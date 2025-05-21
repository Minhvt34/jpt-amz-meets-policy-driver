#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For automatic conversion of C++ containers to Python
#include <string>
#include <vector>
#include <cstdio> // For sscanf if needed, and for FILE*, and printf
#include <cstdlib> // For EXIT_SUCCESS etc.
#include <cmath> // For fabs
#include <climits> // For LLONG_MAX
#include <stdexcept> // For std::runtime_error
#include <cstring>   // For strcpy

namespace py = pybind11;

extern "C" {
    #include "LKH.h"

    extern char *ParameterFileName;
    extern char *ProblemFileName;
    extern char *TourFileName;
    extern char *PiFileName;
    extern char *InitialTourFileName;

    extern double StartTime;
    extern int MaxMatrixDimension;
    extern int CTSPTransform;
    extern int GTSPSets;

    extern int DimensionSaved;
    extern struct Node *NodeSet;
    extern long long MM;
    extern int Precision;

    extern int Norm;
    extern long long BestCost;
    extern long long BestPenalty;
    extern long long CurrentPenalty;
    extern double LowerBound;

    extern int Runs;
    extern int Run;
    extern unsigned int Seed;

    extern double TimeLimit;
    extern int TraceLevel;
    extern int MergingUsed; // From LKHmain logic
    extern int MaxCandidates; // Used for initializing trajectory, needs to be read from params
    extern int *BestTour; // Added for exposing best tour
    extern int Dimension;

    void ReadParameters();
    void ReadProblem();
    void AllocateStructures();
    void CreateCandidateSet();
    void InitializeStatistics();

    long long FindTour();
    void UpdateStatistics(long long Cost, double Time);
    void RecordBetterTour();
    void RecordBestTour();
    void WriteTour(char *FileName, int *Tour, long long Cost);
    void PrintStatistics();
    double GetTime();
    void SRandom(unsigned int Seed);
    long long MergeTourWithBestTour(); // added based on LKHmain logic
    void printff(const char *Format, ...); // added: LKH's printf equivalent

    long long Penalty();
    void StatusReport(long long Cost, double EntryTime, char *Suffix);

    void ChooseInitialTour(); // If used in run_lkh directly or via FindTour
    long long LinKernighan(); // If used in run_lkh directly or via FindTour

    void HashInitialize(HashTable *T); // If hashing is used
}

// Struct to store trajectories for imitation learning (for Python)
// struct LKHTrajectory_py { // Renamed
//     std::vector<LKHState_py> states;
//     std::vector<LKHAction_py> actions;
//     long long final_cost;
//     int dimension;
//     int max_candidates_per_step;
//     size_t recorded_steps;

//     LKHTrajectory_py() : final_cost(0), dimension(0), max_candidates_per_step(0), recorded_steps(0) {}
// };

long long solve_and_record_trajectory(const std::string &param_file, const std::string &problem_file) {
    // Memory management for LKH's global char* filenames
    // LKH doesn't free these, so we manage them here if we set them.
    static std::string last_param_file, last_problem_file;
    static char *param_c_str = nullptr;
    static char *problem_c_str = nullptr;

    // Declare Cost and LastTime here
    long long returned_cost_value; // Renamed to avoid confusion with global BestCost if used later
    double LastTime;

    if (param_c_str) delete[] param_c_str;
    param_c_str = new char[param_file.length() + 1];
    strcpy(param_c_str, param_file.c_str());
    ParameterFileName = param_c_str;

    if (problem_c_str) delete[] problem_c_str;
    problem_c_str = new char[problem_file.length() + 1];
    strcpy(problem_c_str, problem_file.c_str());
    ProblemFileName = problem_c_str; // LKH uses this after ReadParameters too

    printf("PY_WRAP_DEBUG: ParameterFileName set to: %s\n", ParameterFileName);
    printf("PY_WRAP_DEBUG: ProblemFileName set to: %s\n", ProblemFileName);

    // 1. Read LKH parameters (this will set Dimension and MaxCandidates among other things)
    // Initialize a few things LKH main does, just in case
    BestCost = BestPenalty = CurrentPenalty = LLONG_MAX; // Initialize costs
    Runs = 1; // We are doing one run for trajectory recording
    Run = 1; 
    SRandom(1); // Default seed, can be made a parameter

    printf("PY_WRAP_DEBUG: Reading parameters from %s\n", param_file.c_str());
    ReadParameters();

    // 2. Read the TSP problem
    printf("PY_WRAP_DEBUG: Reading problem from %s\n", problem_file.c_str());
    ReadProblem();

    // 3. Allocate memory for LKH structures
    AllocateStructures();

    // 4. Create candidate set
    CreateCandidateSet();

    // 5. Initialize statistics
    InitializeStatistics();

    // 6. Run the solver
    returned_cost_value = FindTour();

    // 7. Record the final tour
    RecordBetterTour();
    RecordBestTour();

    // 8. Get the best tour
    if (Dimension <=0) { // Add a check before trying to new an array based on Dimension
      printff("PY_WRAP_ERROR: Dimension is %d, cannot safely operate on BestTour or BetterTour.\n", Dimension);
      return LLONG_MAX; // Indicate error
    }

    // 9. Return the best tour cost
    return returned_cost_value;
}
PYBIND11_MODULE(lkh_solver, m) {
    m.doc() = "Python bindings for the LKH-AMZ TSP solver";

    m.def("solve_and_record_trajectory", &solve_and_record_trajectory, "Run the LKH-AMZ TSP solver");
}