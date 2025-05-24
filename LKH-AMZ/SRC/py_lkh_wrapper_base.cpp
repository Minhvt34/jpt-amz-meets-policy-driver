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
    extern long long BetterCost;  // Added
    extern long long BestPenalty;
    extern long long BetterPenalty;  // Added
    extern long long CurrentPenalty;
    extern double LowerBound;

    extern int Runs;
    extern int Run;
    extern int Trial;  // Added
    extern int MaxTrials;  // Added
    extern unsigned int Seed;
    extern int HashingUsed;  // Added
    extern HashTable *HTable;  // Added
    
    extern double TimeLimit;
    extern int TraceLevel;
    extern int MergingUsed; // From LKHmain logic
    extern int MaxCandidates; // Used for initializing trajectory, needs to be read from params
    extern int *BestTour; // Added for exposing best tour
    extern int Dimension;
    extern int CandidateSetSymmetric;
    extern double Excess;
    extern int ProblemType;  // Added
    extern struct Node *FirstNode;  // Added
    extern unsigned *Rand;  // Added
    extern unsigned Hash;  // Changed from long long to unsigned to match LKH.h

    void ReadParameters();
    void ReadProblem();
    void AllocateStructures();
    void CreateCandidateSet();
    void InitializeStatistics();
    void AdjustCandidateSet();
    void PrepareKicking();

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

    void GenerateCandidates(int MaxCandidates, long long MaxAlpha, int Symmetric);
    long long Ascent();
    long long Minimum1TreeCost(int Sparse);
    void CandidateReport();

    long long Penalty();
    void StatusReport(long long Cost, double EntryTime, char *Suffix);

    void ChooseInitialTour(); // If used in run_lkh directly or via FindTour
    long long LinKernighan(); // If used in run_lkh directly or via FindTour

    void HashInitialize(HashTable *T); // If hashing is used
    void HashInsert(HashTable *T, unsigned Hash, long long Cost); // Changed second param from long long to unsigned
}

// Function to initialize LKH global variables for a run
void initialize_lkh_run_globals(unsigned int seed_val) {
    BestCost = LLONG_MAX;
    BestPenalty = LLONG_MAX;
    CurrentPenalty = LLONG_MAX;
    Runs = 1; // We are doing one run for trajectory recording
    Run = 1;
    SRandom(seed_val); // Use the provided seed
    
    printf("PY_WRAP_DEBUG: LKH run globals initialized. Seed=%u, BestCost=%lld\n", seed_val, BestCost);
}

// Utility function to check if the solver's internal state is valid and consistent
bool validate_solver_state(bool fix_issues = true) {
    printf("PY_WRAP_DEBUG: Validating solver state...\n");
    
    bool is_valid = true;
    
    // Check if core structures are initialized
    if (NodeSet == nullptr) {
        printf("ERROR: NodeSet is null\n");
        is_valid = false;
    } else {
        printf("PY_WRAP_DEBUG: NodeSet is at %p\n", (void*)NodeSet);
    }
    
    if (Dimension <= 0) {
        printf("ERROR: Dimension is invalid (%d)\n", Dimension);
        is_valid = false;
    } else {
        printf("PY_WRAP_DEBUG: Dimension is %d\n", Dimension);
    }
    
    if (FirstNode == nullptr) {
        printf("ERROR: FirstNode is null\n");
        
        // Try to fix FirstNode if requested and possible
        if (fix_issues && NodeSet != nullptr && Dimension > 0) {
            printf("PY_WRAP_DEBUG: Attempting to initialize FirstNode from NodeSet\n");
            FirstNode = &NodeSet[1]; // First node is typically at index 1 in NodeSet
            
            // Create circular linked list of nodes
            Node *Prev = FirstNode;
            for (int i = 2; i <= Dimension; i++) {
                Node *N = &NodeSet[i];
                N->Pred = Prev;
                Prev->Suc = N;
                Prev = N;
            }
            Prev->Suc = FirstNode;
            FirstNode->Pred = Prev;
            
            printf("PY_WRAP_DEBUG: FirstNode initialized. FirstNode->Id=%d\n", FirstNode->Id);
            is_valid = true; // We've fixed the issue
        } else {
            is_valid = false;
        }
    } else {
        printf("PY_WRAP_DEBUG: FirstNode is at %p, Id=%d\n", (void*)FirstNode, FirstNode->Id);
        
        // Check if the linked list is properly circular
        if (FirstNode->Pred == nullptr || FirstNode->Suc == nullptr) {
            printf("ERROR: FirstNode is not properly linked (Pred=%p, Suc=%p)\n",
                   (void*)FirstNode->Pred, (void*)FirstNode->Suc);
            
            // Try to fix the circular linked list if requested and possible
            if (fix_issues && NodeSet != nullptr && Dimension > 0) {
                printf("PY_WRAP_DEBUG: Attempting to repair node linkage\n");
                Node *Prev = FirstNode;
                for (int i = 2; i <= Dimension; i++) {
                    Node *N = &NodeSet[i];
                    N->Pred = Prev;
                    Prev->Suc = N;
                    Prev = N;
                }
                Prev->Suc = FirstNode;
                FirstNode->Pred = Prev;
                printf("PY_WRAP_DEBUG: Node linkage repaired\n");
                is_valid = true; // We've fixed the issue
            } else {
                is_valid = false;
            }
        }
    }
    
    printf("PY_WRAP_DEBUG: Solver state validation %s\n", is_valid ? "passed" : "failed");
    return is_valid;
}

// =====================================================================================
// Custom implementation of CreateCandidateSet with explicit parameters
// =====================================================================================

bool create_candidate_set_explicit() {
    try {
        // Check global variables before we start
        if (FirstNode == nullptr) {
            printf("ERROR: FirstNode is null, cannot create candidate set\n");
            return false;
        }
        if (Dimension <= 0) {
            printf("ERROR: Dimension is invalid (%d), cannot create candidate set\n", Dimension);
            return false;
        }
        if (MaxCandidates < 0) {
            printf("ERROR: MaxCandidates is invalid (%d), cannot create candidate set\n", MaxCandidates);
            return false;
        }
        
        printf("Creating candidates explicitly...\n");
        
        // Additional checks on FirstNode structure
        printf("PY_WRAP_DEBUG: FirstNode->Id=%d, FirstNode->Pred=%p, FirstNode->Suc=%p\n", 
               FirstNode->Id, (void*)FirstNode->Pred, (void*)FirstNode->Suc);
               
        // Check if the linked list is properly circular
        if (FirstNode->Pred == nullptr || FirstNode->Suc == nullptr) {
            printf("ERROR: FirstNode is not properly linked (Pred=%p, Suc=%p)\n",
                   (void*)FirstNode->Pred, (void*)FirstNode->Suc);
                   
            // Attempt to fix the circular linked list if needed
            if (NodeSet != nullptr && Dimension > 0) {
                printf("PY_WRAP_DEBUG: Attempting to repair node linkage\n");
                Node *Prev = FirstNode;
                for (int i = 2; i <= Dimension; i++) {
                    Node *N = &NodeSet[i];
                    N->Pred = Prev;
                    Prev->Suc = N;
                    Prev = N;
                }
                Prev->Suc = FirstNode;
                FirstNode->Pred = Prev;
                printf("PY_WRAP_DEBUG: Node linkage repaired\n");
            } else {
                return false;
            }
        }
        
        // Implement the functionality of CreateCandidateSet directly
        long long Cost, MaxAlpha;
        Node *Na;
        int i;
        double EntryTime = GetTime();

        Norm = 9999;
        if (C == C_EXPLICIT) {
            printf("Processing C_EXPLICIT - scaling costs by precision\n");
            Na = FirstNode;
            do {
                for (i = 1; i < Na->Id; i++)
                    Na->C[i] *= Precision;
            } while ((Na = Na->Suc) != FirstNode);
        }
        
        printf("Setting Pi values to 0\n");
        Na = FirstNode;
        do
            Na->Pi = 0;
        while ((Na = Na->Suc) != FirstNode);
        
        printf("Computing Ascent\n");
        Cost = Ascent();
        
        if (MaxCandidates > 0) {
            printf("Computing Minimum1TreeCost (sparse=0)\n");
            Cost = Minimum1TreeCost(0);
        } else {
            printf("Computing Minimum1TreeCost (sparse=1)\n");
            Cost = Minimum1TreeCost(1);
        }
        
        printf("Setting LowerBound\n");
        LowerBound = (double) Cost / Precision;
        
        printf("Computing MaxAlpha\n");
        MaxAlpha = (long long) fabs(Excess * Cost);
        
        printf("Generating candidates (MaxCandidates=%d, MaxAlpha=%lld, Symmetric=%d)\n", 
               MaxCandidates, MaxAlpha, CandidateSetSymmetric);
        GenerateCandidates(MaxCandidates, MaxAlpha, CandidateSetSymmetric);

        // Validation step from original function
        if (MaxTrials > 0) {
            printf("Validating that each node has candidates\n");
            Na = FirstNode;
            do {
                if (!Na->CandidateSet || !Na->CandidateSet[0].To) {
                    if (MaxCandidates == 0) {
                        printf("ERROR: MAX_CANDIDATES = 0: Node %d has no candidates\n", Na->Id);
                        return false;
                    } else {
                        printf("ERROR: Node %d has no candidates\n", Na->Id);
                        return false;
                    }
                }
            } while ((Na = Na->Suc) != FirstNode);
        }
        
        // Finalize like the original function
        if (C == C_EXPLICIT) {
            printf("Finalizing for C_EXPLICIT\n");
            Na = FirstNode;
            do
                for (i = 1; i < Na->Id; i++)
                    Na->C[i] += Na->Pi + NodeSet[i].Pi;
            while ((Na = Na->Suc) != FirstNode);
        }
        
        // Output report
        CandidateReport();
        printf("CreateCandidateSet completed in %.2f sec\n", fabs(GetTime() - EntryTime));
        
        return true;
    }
    catch (const std::exception& e) {
        printf("Exception in create_candidate_set_explicit: %s\n", e.what());
        return false;
    }
    catch (...) {
        printf("Unknown exception in create_candidate_set_explicit\n");
        return false;
    }
}

// =====================================================================================
// Original safe wrapper functions for LKH functions
// =====================================================================================

// Safe wrapper for CreateCandidateSet - handles global variables properly
bool safe_create_candidate_set() {
    try {
        // The new explicit implementation is safer
        return create_candidate_set_explicit();
    }
    catch (const std::exception& e) {
        printf("Exception in safe_create_candidate_set: %s\n", e.what());
        return false;
    }
    catch (...) {
        printf("Unknown exception in safe_create_candidate_set\n");
        return false;
    }
}

// Safe wrapper for LinKernighan - handles global variables
long long safe_lin_kernighan() {
    try {
        // Validate required global variables
        if (FirstNode == nullptr) {
            printf("ERROR: FirstNode is null, cannot run LinKernighan\n");
            return LLONG_MAX;
        }

        // Call the original function
        return LinKernighan();
    }
    catch (const std::exception& e) {
        printf("Exception in safe_lin_kernighan: %s\n", e.what());
        return LLONG_MAX;
    }
    catch (...) {
        printf("Unknown exception in safe_lin_kernighan\n");
        return LLONG_MAX;
    }
}

// Safe wrapper for ChooseInitialTour - handles global variables
bool safe_choose_initial_tour() {
    try {
        // Validate required global variables
        if (FirstNode == nullptr) {
            printf("ERROR: FirstNode is null, cannot choose initial tour\n");
            return false;
        }

        // Call the original function
        ChooseInitialTour();
        return true;
    }
    catch (const std::exception& e) {
        printf("Exception in safe_choose_initial_tour: %s\n", e.what());
        return false;
    }
    catch (...) {
        printf("Unknown exception in safe_choose_initial_tour\n");
        return false;
    }
}

// Safe wrapper for RecordBetterTour - handles global variables
bool safe_record_better_tour() {
    try {
        // Validate required global variables
        if (FirstNode == nullptr) {
            printf("ERROR: FirstNode is null, cannot record better tour\n");
            return false;
        }

        // Call the original function
        RecordBetterTour();
        return true;
    }
    catch (const std::exception& e) {
        printf("Exception in safe_record_better_tour: %s\n", e.what());
        return false;
    }
    catch (...) {
        printf("Unknown exception in safe_record_better_tour\n");
        return false;
    }
}

// Safe wrapper for AdjustCandidateSet - handles global variables
bool safe_adjust_candidate_set() {
    try {
        // Validate required global variables
        if (FirstNode == nullptr) {
            printf("ERROR: FirstNode is null, cannot adjust candidate set\n");
            return false;
        }

        // Call the original function
        AdjustCandidateSet();
        return true;
    }
    catch (const std::exception& e) {
        printf("Exception in safe_adjust_candidate_set: %s\n", e.what());
        return false;
    }
    catch (...) {
        printf("Unknown exception in safe_adjust_candidate_set\n");
        return false;
    }
}

// Safe wrapper for PrepareKicking - handles global variables
bool safe_prepare_kicking() {
    try {
        // Validate required global variables
        if (FirstNode == nullptr) {
            printf("ERROR: FirstNode is null, cannot prepare kicking\n");
            return false;
        }

        // Call the original function
        PrepareKicking();
        return true;
    }
    catch (const std::exception& e) {
        printf("Exception in safe_prepare_kicking: %s\n", e.what());
        return false;
    }
    catch (...) {
        printf("Unknown exception in safe_prepare_kicking\n");
        return false;
    }
}

// Safe wrapper for RecordBestTour - handles global variables
bool safe_record_best_tour() {
    try {
        // Validate required global variables
        if (FirstNode == nullptr) {
            printf("ERROR: FirstNode is null, cannot record best tour\n");
            return false;
        }

        // Call the original function
        RecordBestTour();
        return true;
    }
    catch (const std::exception& e) {
        printf("Exception in safe_record_best_tour: %s\n", e.what());
        return false;
    }
    catch (...) {
        printf("Unknown exception in safe_record_best_tour\n");
        return false;
    }
}

// Function to access the BestTour array and return it as a vector
std::vector<int> get_best_tour() {
    if (BestTour == nullptr || Dimension <= 0) {
        throw std::runtime_error("BestTour is not available or Dimension is invalid");
    }

    // Create a vector to hold the tour
    std::vector<int> tour;
    
    // Copy the tour values
    // Best tour in LKH is stored from 1 to Dimension (1-indexed), and the first node is repeated at the end
    for (int i = 1; i <= Dimension + 1; i++) {
        tour.push_back(BestTour[i]);
    }
    
    return tour;
}

// Function to get the current dimension
int get_dimension() {
    return Dimension;
}

// Function to get the best cost
long long get_best_cost() {
    return BestCost;
}

void read_problem_file(const std::string &problem_file) {
    // Memory management for LKH's global char* filenames
    static char *problem_c_str = nullptr;
    
    if (problem_c_str) delete[] problem_c_str;
    problem_c_str = new char[problem_file.length() + 1];
    strcpy(problem_c_str, problem_file.c_str());
    ProblemFileName = problem_c_str; // LKH uses this after ReadParameters too
}

void read_parameter_file(const std::string &param_file) {
    // Memory management for LKH's global char* filenames
    static char *param_c_str = nullptr;
    
    if (param_c_str) delete[] param_c_str;
    param_c_str = new char[param_file.length() + 1];
    strcpy(param_c_str, param_file.c_str());
    ParameterFileName = param_c_str;
}

long long solve_and_record_trajectory(const std::string &param_file, const std::string &problem_file) {
    // Memory management for LKH's global char* filenames
    // LKH doesn't free these, so we manage them here if we set them.
    static std::string last_param_file, last_problem_file;
    static char *param_c_str = nullptr;
    static char *problem_c_str = nullptr;
    
    // Static buffer for empty string parameter
    static char empty_str[] = "";

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

    // Initialize variables for LKH
    BestCost = BestPenalty = CurrentPenalty = LLONG_MAX; // Initialize costs
    Runs = 1; // We are doing one run for trajectory recording
    Run = 1; 
    SRandom(1); // Default seed, can be made a parameter

    // 1. Read LKH parameters
    printf("PY_WRAP_DEBUG: Reading parameters from %s\n", param_file.c_str());
    ReadParameters();
    printf("PY_WRAP_DEBUG: Parameters read. MaxCandidates=%d, TraceLevel=%d\n", 
           MaxCandidates, TraceLevel);

    // 2. Read the TSP problem
    printf("PY_WRAP_DEBUG: Reading problem from %s\n", problem_file.c_str());
    ReadProblem();
    printf("PY_WRAP_DEBUG: Problem read. Dimension=%d, ProblemType=%d\n", 
           Dimension, ProblemType);

    // 3. Allocate memory for LKH structures
    printf("PY_WRAP_DEBUG: Allocating structures\n");
    AllocateStructures();
    
    // Validate solver state after initialization
    if (!validate_solver_state(true)) {
        printf("ERROR: Solver state validation failed after initialization\n");
        return LLONG_MAX;
    }

    // 4. Create candidate set
    printf("PY_WRAP_DEBUG: Creating candidate set\n");
    if (!create_candidate_set_explicit()) {
        printf("ERROR: Failed to create candidate set in solve_and_record_trajectory\n");
        return LLONG_MAX;
    }

    // 5. Initialize statistics
    printf("PY_WRAP_DEBUG: Initializing statistics\n");
    InitializeStatistics();
    
    // Validate solver state again before running the algorithm
    if (!validate_solver_state(true)) {
        printf("ERROR: Solver state validation failed before running algorithm\n");
        return LLONG_MAX;
    }

    // 6. Run the solver
    //returned_cost_value = FindTour();
    long long Cost_2;
    Node *t;
    int i2;
    double EntryTime_2 = GetTime();

    printf("PY_WRAP_DEBUG: Initializing nodes for solver run\n");
    t = FirstNode;
    do
        t->OldPred = t->OldSuc = t->NextBestSuc = t->BestSuc = 0;
    while ((t = t->Suc) != FirstNode);
    
    BetterCost = LLONG_MAX;
    BetterPenalty = CurrentPenalty = LLONG_MAX;
    
    if (MaxTrials > 0) {
        printf("PY_WRAP_DEBUG: Using MaxTrials = %d\n", MaxTrials);
        if (HashingUsed)
            HashInitialize(HTable);
    } else {
        printf("PY_WRAP_DEBUG: MaxTrials = 0, choosing initial tour directly\n");
        Trial = 1;
        ChooseInitialTour();
        CurrentPenalty = LLONG_MAX;
        CurrentPenalty = BetterPenalty = Penalty();
    }
    
    printf("PY_WRAP_DEBUG: Preparing kicking\n");
    PrepareKicking();
    
    printf("PY_WRAP_DEBUG: Starting trials loop (MaxTrials=%d)\n", MaxTrials);
    for (Trial = 1; Trial <= MaxTrials; Trial++) {
        printf("PY_WRAP_DEBUG: Trial %d/%d\n", Trial, MaxTrials);
        
        if (Trial > 1 && GetTime() - StartTime >= TimeLimit) {
            printf("PY_WRAP_DEBUG: Time limit exceeded\n");
            if (TraceLevel >= 1)
                printff("*** Time limit exceeded ***\n");
            break;
        }
        
        /* Choose FirstNode at random */
        if (Dimension == DimensionSaved) {
            FirstNode = &NodeSet[1 + Random() % Dimension];
            printf("PY_WRAP_DEBUG: FirstNode randomly chosen: %d\n", FirstNode->Id);
        } else {
            for (i2 = Random() % Dimension; i2 > 0; i2--)
                FirstNode = FirstNode->Suc;
            printf("PY_WRAP_DEBUG: FirstNode set to: %d\n", FirstNode->Id);
        }
        
        printf("PY_WRAP_DEBUG: Choosing initial tour\n");
        ChooseInitialTour();
        
        // Validate solver state before LinKernighan
        if (!validate_solver_state(true)) {
            printf("ERROR: Solver state validation failed before LinKernighan at trial %d\n", Trial);
            continue; // Skip this trial if the state is invalid
        }
        
        printf("PY_WRAP_DEBUG: Running LinKernighan\n");
        Cost_2 = LinKernighan();
        printf("PY_WRAP_DEBUG: LinKernighan completed with cost %lld\n", Cost_2);
        
        if (CurrentPenalty < BetterPenalty ||
            (CurrentPenalty == BetterPenalty && Cost_2 < BetterCost)) {
            if (TraceLevel >= 1) {
                printff("* %d: ", Trial);
                StatusReport(Cost_2, EntryTime_2, empty_str);
            }
            BetterCost = Cost_2;
            BetterPenalty = CurrentPenalty;
            
            printf("PY_WRAP_DEBUG: Recording better tour\n");
            RecordBetterTour();
            
            printf("PY_WRAP_DEBUG: Adjusting candidate set\n");
            AdjustCandidateSet();
            
            printf("PY_WRAP_DEBUG: Preparing kicking\n");
            PrepareKicking();
            
            if (HashingUsed) {
                HashInitialize(HTable);
                HashInsert(HTable, Hash, Cost_2);
            }
        } else if (TraceLevel >= 2) {
            printff("  %d: ", Trial);
            StatusReport(Cost_2, EntryTime_2, empty_str);
        }
    }
    
    printf("PY_WRAP_DEBUG: Trials complete, finalizing tour\n");
    t = FirstNode;
    if (Norm == 0 || MaxTrials == 0 || !t->BestSuc) {
        do
            t = t->BestSuc = t->Suc;
        while (t != FirstNode);
    }
    do
        (t->Suc = t->BestSuc)->Pred = t;
    while ((t = t->BestSuc) != FirstNode);
    
    if (HashingUsed) {
        Hash = 0;
        do
            Hash ^= Rand[t->Id] * Rand[t->Suc->Id];
        while ((t = t->BestSuc) != FirstNode);
    }
    
    if (Trial > MaxTrials)
        Trial = MaxTrials;
    CurrentPenalty = BetterPenalty;

    // 7. Record the final tour
    printf("PY_WRAP_DEBUG: Recording best tour\n");
    RecordBestTour();

    // 8. Get the best tour
    if (Dimension <=0) { // Add a check before trying to new an array based on Dimension
      printff("PY_WRAP_ERROR: Dimension is %d, cannot safely operate on BestTour or BetterTour.\n", Dimension);
      return LLONG_MAX; // Indicate error
    }
    
    printf("PY_WRAP_DEBUG: Best tour: ");
    for (int i = 0; i <= DimensionSaved && i < 10; i++) { // Print just the first 10 nodes to avoid clutter
        printf("%d ", BestTour[i]);
    }
    printf("... (truncated for brevity)\n");

    // 9. Return the best tour cost
    printf("PY_WRAP_DEBUG: Returning best cost: %lld\n", BetterCost);
    return BetterCost;
}

// =====================================================================================
// New Python-STEP-Solver helper functions (Granular Control)
// =====================================================================================

// Node State Management
void py_reset_node_tour_fields() {
    Node *t = FirstNode;
    if (!t) {
        printf("PY_WRAP_ERROR: FirstNode is NULL in py_reset_node_tour_fields\n");
        return;
    }
    do {
        t->OldPred = t->OldSuc = t->NextBestSuc = t->BestSuc = 0;
    } while ((t = t->Suc) != FirstNode);
    printf("PY_WRAP_DEBUG: Node tour fields reset (OldPred, OldSuc, NextBestSuc, BestSuc).\n");
}

// Cost and Penalty Getters/Setters
long long get_better_cost() { return BetterCost; }
void set_better_cost(long long cost) { 
    BetterCost = cost;
    printf("PY_WRAP_DEBUG: BetterCost set to %lld\n", BetterCost);
}

long long get_better_penalty() { return BetterPenalty; }
void set_better_penalty(long long penalty) { 
    BetterPenalty = penalty;
    printf("PY_WRAP_DEBUG: BetterPenalty set to %lld\n", BetterPenalty);
}

long long get_current_penalty() { return CurrentPenalty; }
void set_current_penalty(long long penalty) { 
    CurrentPenalty = penalty; 
    printf("PY_WRAP_DEBUG: CurrentPenalty set to %lld\n", CurrentPenalty);
}

// FirstNode Selection
void py_select_random_first_node() {
    if (Dimension <= 0) {
        printf("PY_WRAP_ERROR: Dimension invalid in py_select_random_first_node\n");
        return;
    }
    if (Dimension == DimensionSaved) {
        FirstNode = &NodeSet[1 + Random() % Dimension];
    } else {
        // This case (Dimension != DimensionSaved) is complex and usually related to problem transformations.
        // For typical TSP, Dimension == DimensionSaved after ReadProblem.
        // If it occurs, ensure FirstNode is already part of a valid circular list.
        if (!FirstNode) { 
            printf("PY_WRAP_ERROR: FirstNode is NULL before random selection (Dim != DimSaved)\n"); 
            return; 
        }
        for (int i = Random() % Dimension; i > 0; i--)
            FirstNode = FirstNode->Suc;
    }
    if(FirstNode) printf("PY_WRAP_DEBUG: Random FirstNode selected: ID %d\n", FirstNode->Id);
    else printf("PY_WRAP_ERROR: FirstNode became NULL after selection\n");

}

int get_first_node_id() {
    if (FirstNode) return FirstNode->Id;
    return -1; // Error or not set
}

// Trial Management
void set_trial_number(int n) { 
    Trial = n; 
    // printf("PY_WRAP_DEBUG: Trial set to %d\n", Trial); // Can be verbose
}
int get_trial_number() { return Trial; }

// Hashing Control
bool is_hashing_used() { return (bool)HashingUsed; }
unsigned get_lkh_hash() { return Hash; } // LKH global Hash

// Tour Finalization
void py_finalize_tour_from_best_suc() {
    Node *t = FirstNode;
    if (!t) {
        printf("PY_WRAP_ERROR: FirstNode is NULL in py_finalize_tour_from_best_suc\n");
        return;
    }
    printf("PY_WRAP_DEBUG: Finalizing tour from BestSuc chain...\n");

    // This logic mirrors the end of LKH's FindTour and the C++ solve_and_record_trajectory
    if (Norm == 0 || MaxTrials == 0 || !t->BestSuc) {
        printf("PY_WRAP_DEBUG: Setting BestSuc = Suc for all nodes as fallback/initial state in finalize.\n");
        Node* current = FirstNode;
        do {
            if (!current->BestSuc) current->BestSuc = current->Suc;
            current = current->Suc;
        } while (current != FirstNode);
    }
    
    t = FirstNode;
    do {
        if (!t->BestSuc) { 
            printf("PY_WRAP_WARNING: Node %d BestSuc is NULL during finalization. Using t->Suc.\n", t->Id);
            t->BestSuc = t->Suc; 
        }
        (t->Suc = t->BestSuc)->Pred = t;
    } while ((t = t->BestSuc) != FirstNode);
    printf("PY_WRAP_DEBUG: Suc pointers updated from BestSuc chain.\n");

    if (HashingUsed) {
        Hash = 0; // Recalculate hash based on the final tour
        t = FirstNode;
        do {
            Hash ^= Rand[t->Id] * Rand[t->Suc->Id];
        } while ((t = t->Suc) != FirstNode);
        printf("PY_WRAP_DEBUG: Final Hash recalculated: %u\n", Hash);
    }
}

// Expose Penalty() function
long long py_calculate_penalty() {
    long long p = Penalty();
    printf("PY_WRAP_DEBUG: Penalty() called, result: %lld\n", p);
    return p;
}

// Wrapper for HashInitialize to be called from Python without args
void py_wrapper_hash_initialize() {
    if (HTable) { // Ensure HTable is not null, though it should be allocated by AllocateStructures if HashingUsed
        HashInitialize(HTable);
        printf("PY_WRAP_DEBUG: HashInitialize(HTable) called via wrapper.\n");
    } else {
        printf("PY_WRAP_ERROR: HTable is NULL in py_wrapper_hash_initialize. Hashing might not be properly set up.\n");
    }
}

PYBIND11_MODULE(lkh_solver, m) {
    m.doc() = "Python bindings for the LKH-AMZ TSP solver";

    m.def("solve_and_record_trajectory", &solve_and_record_trajectory, "Run the LKH-AMZ TSP solver");
    m.def("read_problem_file", &read_problem_file, "Read the TSP problem file");
    m.def("read_parameter_file", &read_parameter_file, "Read the LKH parameters file");
    m.def("AllocateStructures", &AllocateStructures, "Allocate memory for LKH structures");
    
    // Replace direct C function bindings with safe wrappers
    m.def("CreateCandidateSet", &safe_create_candidate_set, "Safely create the candidate set");
    m.def("ChooseInitialTour", &safe_choose_initial_tour, "Safely choose the initial tour");
    m.def("LinKernighan", &safe_lin_kernighan, "Safely run the Lin-Kernighan algorithm");
    m.def("RecordBetterTour", &safe_record_better_tour, "Safely record the better tour");
    m.def("RecordBestTour", &safe_record_best_tour, "Safely record the best tour");
    m.def("AdjustCandidateSet", &safe_adjust_candidate_set, "Safely adjust the candidate set");
    m.def("PrepareKicking", &safe_prepare_kicking, "Safely prepare for kicking");
    
    m.def("InitializeStatistics", &InitializeStatistics, "Initialize statistics");
    m.def("UpdateStatistics", &UpdateStatistics, "Update statistics");
    m.def("StatusReport", &StatusReport, "Report the status");
    m.def("Penalty", &Penalty, "Calculate the penalty"); // Direct Penalty, py_calculate_penalty is a verbose wrapper
    
    // Modified HashInitialize binding to use the wrapper
    m.def("HashInitialize", &py_wrapper_hash_initialize, "Initialize the hash table (uses global HTable)"); 
    m.def("HashInsert", &HashInsert, "Insert into the hash table");
    
    // Add the new functions
    m.def("get_best_tour", &get_best_tour, "Get the best tour found by the solver");
    m.def("get_dimension", &get_dimension, "Get the dimension of the problem");
    m.def("get_best_cost", &get_best_cost, "Get the cost of the best tour");
    
    // Add debugging/validation functions
    m.def("validate_solver_state", &validate_solver_state, py::arg("fix_issues") = true,
          "Validate the solver's internal state and optionally fix issues");

    // Expose LKH's core file reading functions
    m.def("LKH_ReadParameters", &ReadParameters, "Invoke LKH's internal ReadParameters function");
    m.def("LKH_ReadProblem", &ReadProblem, "Invoke LKH's internal ReadProblem function");

    // Expose initialization function
    m.def("initialize_lkh_run_globals", &initialize_lkh_run_globals, "Initialize LKH global variables for a run");
    m.def("SRandom", &SRandom, "Set the seed for LKH's random number generator");

    // Granular control functions for Python-driven solver logic
    m.def("py_reset_node_tour_fields", &py_reset_node_tour_fields, "Resets OldPred, OldSuc, NextBestSuc, BestSuc for all nodes");
    m.def("get_better_cost", &get_better_cost, "Get current BetterCost");
    m.def("set_better_cost", &set_better_cost, "Set current BetterCost");
    m.def("get_better_penalty", &get_better_penalty, "Get current BetterPenalty");
    m.def("set_better_penalty", &set_better_penalty, "Set current BetterPenalty");
    m.def("get_current_penalty", &get_current_penalty, "Get current CurrentPenalty (often set by LK or Penalty())");
    m.def("set_current_penalty", &set_current_penalty, "Set current CurrentPenalty");
    m.def("py_select_random_first_node", &py_select_random_first_node, "Selects a random FirstNode for a trial");
    m.def("get_first_node_id", &get_first_node_id, "Gets the ID of the current FirstNode");
    m.def("set_trial_number", &set_trial_number, "Set LKH global Trial number");
    m.def("get_trial_number", &get_trial_number, "Get LKH global Trial number");
    m.def("is_hashing_used", &is_hashing_used, "Checks if HashingUsed is enabled in LKH");
    m.def("get_lkh_hash", &get_lkh_hash, "Gets the current LKH global Hash value");
    m.def("py_finalize_tour_from_best_suc", &py_finalize_tour_from_best_suc, "Finalizes tour by setting Suc from BestSuc and recalculates Hash");
    m.def("py_calculate_penalty", &py_calculate_penalty, "Calls LKH Penalty() function and returns its result");

    // Note: HashInitialize and HashInsert are already exposed.
    // Safe wrappers for LKH algorithmic steps like RecordBetterTour, AdjustCandidateSet, PrepareKicking,
    // ChooseInitialTour, LinKernighan, RecordBestTour remain essential.
}