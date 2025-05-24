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
#include <memory>    // For smart pointers
#include <mutex>     // For thread safety

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

// Global mutex for thread safety when accessing LKH globals
static std::mutex lkh_global_mutex;

/**
 * LKHSolver class - Encapsulates all LKH state in a separate instance
 * This allows multiple independent solver instances for multiprocessing
 */
class LKHSolver {
private:
    // File paths - managed by this instance
    std::string param_file_path;
    std::string problem_file_path;
    std::string tour_file_path;
    std::string pi_file_path;
    std::string initial_tour_file_path;
    
    // C-style strings for LKH functions (allocated/deallocated by this instance)
    char* param_file_cstr;
    char* problem_file_cstr;
    char* tour_file_cstr;
    char* pi_file_cstr;
    char* initial_tour_file_cstr;
    
    // Instance-specific copies of LKH global variables
    double instance_StartTime;
    int instance_MaxMatrixDimension;
    int instance_CTSPTransform;
    int instance_GTSPSets;
    int instance_DimensionSaved;
    
    struct Node *instance_NodeSet;
    long long instance_MM;
    int instance_Precision;
    int instance_Norm;
    
    long long instance_BestCost;
    long long instance_BetterCost;
    long long instance_BestPenalty;
    long long instance_BetterPenalty;
    long long instance_CurrentPenalty;
    double instance_LowerBound;
    
    int instance_Runs;
    int instance_Run;
    int instance_Trial;
    int instance_MaxTrials;
    unsigned int instance_Seed;
    int instance_HashingUsed;
    HashTable *instance_HTable;
    
    double instance_TimeLimit;
    int instance_TraceLevel;
    int instance_MergingUsed;
    int instance_MaxCandidates;
    int *instance_BestTour;
    int instance_Dimension;
    int instance_CandidateSetSymmetric;
    double instance_Excess;
    int instance_ProblemType;
    struct Node *instance_FirstNode;
    unsigned *instance_Rand;
    unsigned instance_Hash;
    
    // State management
    bool initialized;
    bool problem_loaded;
    bool structures_allocated;
    
    // Helper methods for state management
    void install_globals();
    void uninstall_globals();
    void copy_globals_to_instance();
    void allocate_cstrings();
    void deallocate_cstrings();
    
    // Internal helper methods (called by public methods or solve_with_trajectory)
    // These assume the global lock is held and LKH globals are installed.
    bool read_parameters_internal();
    bool read_problem_internal();
    bool allocate_structures_internal();
    bool create_candidate_set_internal();
    void initialize_statistics_internal();
    bool choose_initial_tour_internal();
    long long lin_kernighan_internal();
    bool record_better_tour_internal();
    bool record_best_tour_internal();
    bool adjust_candidate_set_internal();
    bool prepare_kicking_internal();
    void reset_node_tour_fields_internal();
    void select_random_first_node_internal();
    void finalize_tour_from_best_suc_internal();
    long long calculate_penalty_internal();
    void hash_initialize_internal();
    void hash_insert_internal(unsigned hash_val, long long cost);
    long long calculate_tour_cost_internal();
    // Note: initialize_run_globals_internal is not strictly needed as the public one handles it well.
    
public:
    // Constructor/Destructor
    LKHSolver();
    ~LKHSolver();
    
    // File management
    void set_parameter_file(const std::string& filename);
    void set_problem_file(const std::string& filename);
    void set_tour_file(const std::string& filename);
    void set_pi_file(const std::string& filename);
    void set_initial_tour_file(const std::string& filename);
    
    // Initialization
    bool read_parameters();
    bool read_problem();
    bool allocate_structures();
    void initialize_run_globals(unsigned int seed_val);
    
    // Core solver functions
    bool create_candidate_set();
    void initialize_statistics();
    bool choose_initial_tour();
    long long lin_kernighan();
    bool record_better_tour();
    bool record_best_tour();
    bool adjust_candidate_set();
    bool prepare_kicking();
    
    // State management functions
    void reset_node_tour_fields();
    void select_random_first_node();
    void finalize_tour_from_best_suc();
    long long calculate_penalty();
    
    // Hash functions
    void hash_initialize();
    void hash_insert(unsigned hash_val, long long cost);
    
    // Getters and setters
    long long get_best_cost() const { return instance_BestCost; }
    long long get_better_cost() const { return instance_BetterCost; }
    long long get_better_penalty() const { return instance_BetterPenalty; }
    long long get_current_penalty() const { return instance_CurrentPenalty; }
    int get_dimension() const { return instance_Dimension; }
    int get_first_node_id() const;
    int get_trial_number() const { return instance_Trial; }
    bool is_hashing_used() const { return instance_HashingUsed != 0; }
    unsigned get_lkh_hash() const { return instance_Hash; }
    
    void set_better_cost(long long cost) { instance_BetterCost = cost; }
    void set_better_penalty(long long penalty) { instance_BetterPenalty = penalty; }
    void set_current_penalty(long long penalty) { instance_CurrentPenalty = penalty; }
    void set_trial_number(int trial) { instance_Trial = trial; }
    
    // Tour access
    std::vector<int> get_best_tour();
    
    // High-level solver interface
    long long solve_with_trajectory(int max_trials = 10, double time_limit = 3600.0);
    
    // Validation
    bool validate_solver_state(bool fix_issues = true);
    
    // Calculate actual tour cost from current tour structure
    long long calculate_tour_cost();
};

// Implementation of LKHSolver methods

LKHSolver::LKHSolver() 
    : param_file_cstr(nullptr), problem_file_cstr(nullptr), tour_file_cstr(nullptr),
      pi_file_cstr(nullptr), initial_tour_file_cstr(nullptr),
      instance_NodeSet(nullptr), instance_HTable(nullptr), instance_BestTour(nullptr),
      instance_FirstNode(nullptr), instance_Rand(nullptr),
      initialized(false), problem_loaded(false), structures_allocated(false) {
    
    // Initialize instance variables with default values
    instance_StartTime = 0.0;
    instance_MaxMatrixDimension = 0;
    instance_CTSPTransform = 0;
    instance_GTSPSets = 0;
    instance_DimensionSaved = 0;
    instance_MM = 0;
    instance_Precision = 1;
    instance_Norm = 0;
    
    instance_BestCost = LLONG_MAX;
    instance_BetterCost = LLONG_MAX;
    instance_BestPenalty = LLONG_MAX;
    instance_BetterPenalty = LLONG_MAX;
    instance_CurrentPenalty = LLONG_MAX;
    instance_LowerBound = 0.0;
    
    instance_Runs = 1;
    instance_Run = 1;
    instance_Trial = 1;
    instance_MaxTrials = 0;
    instance_Seed = 1;
    instance_HashingUsed = 0;
    
    instance_TimeLimit = 3600.0;
    instance_TraceLevel = 0;
    instance_MergingUsed = 0;
    instance_MaxCandidates = 0;
    instance_Dimension = 0;
    instance_CandidateSetSymmetric = 0;
    instance_Excess = 0.0;
    instance_ProblemType = 0;
    instance_Hash = 0;
}

LKHSolver::~LKHSolver() {
    deallocate_cstrings();
    // Note: LKH structures are typically managed by LKH itself
    // but we might need to add cleanup for instance-specific allocations
}

void LKHSolver::allocate_cstrings() {
    deallocate_cstrings(); // Clean up any existing allocations
    
    if (!param_file_path.empty()) {
        param_file_cstr = new char[param_file_path.length() + 1];
        strcpy(param_file_cstr, param_file_path.c_str());
    }
    
    if (!problem_file_path.empty()) {
        problem_file_cstr = new char[problem_file_path.length() + 1];
        strcpy(problem_file_cstr, problem_file_path.c_str());
    }
    
    if (!tour_file_path.empty()) {
        tour_file_cstr = new char[tour_file_path.length() + 1];
        strcpy(tour_file_cstr, tour_file_path.c_str());
    }
    
    if (!pi_file_path.empty()) {
        pi_file_cstr = new char[pi_file_path.length() + 1];
        strcpy(pi_file_cstr, pi_file_path.c_str());
    }
    
    if (!initial_tour_file_path.empty()) {
        initial_tour_file_cstr = new char[initial_tour_file_path.length() + 1];
        strcpy(initial_tour_file_cstr, initial_tour_file_path.c_str());
    }
}

void LKHSolver::deallocate_cstrings() {
    delete[] param_file_cstr; param_file_cstr = nullptr;
    delete[] problem_file_cstr; problem_file_cstr = nullptr;
    delete[] tour_file_cstr; tour_file_cstr = nullptr;
    delete[] pi_file_cstr; pi_file_cstr = nullptr;
    delete[] initial_tour_file_cstr; initial_tour_file_cstr = nullptr;
}

void LKHSolver::install_globals() {
    // Install instance variables into LKH globals
    // This is called before any LKH function that uses globals
    extern char *ParameterFileName, *ProblemFileName, *TourFileName, *PiFileName, *InitialTourFileName;
    extern double StartTime;
    extern int MaxMatrixDimension, CTSPTransform, GTSPSets, DimensionSaved;
    extern struct Node *NodeSet, *FirstNode;
    extern long long MM, BestCost, BetterCost, BestPenalty, BetterPenalty, CurrentPenalty;
    extern int Precision, Norm, Runs, Run, Trial, MaxTrials, HashingUsed;
    extern unsigned int Seed;
    extern HashTable *HTable;
    extern double TimeLimit, LowerBound;
    extern int TraceLevel, MergingUsed, MaxCandidates, Dimension, CandidateSetSymmetric, ProblemType;
    extern int *BestTour;
    extern double Excess;
    extern unsigned *Rand;
    extern unsigned Hash;
    
    ParameterFileName = param_file_cstr;
    ProblemFileName = problem_file_cstr;
    TourFileName = tour_file_cstr;
    PiFileName = pi_file_cstr;
    InitialTourFileName = initial_tour_file_cstr;
    
    StartTime = instance_StartTime;
    MaxMatrixDimension = instance_MaxMatrixDimension;
    CTSPTransform = instance_CTSPTransform;
    GTSPSets = instance_GTSPSets;
    DimensionSaved = instance_DimensionSaved;
    
    NodeSet = instance_NodeSet;
    FirstNode = instance_FirstNode;
    MM = instance_MM;
    Precision = instance_Precision;
    Norm = instance_Norm;
    
    BestCost = instance_BestCost;
    BetterCost = instance_BetterCost;
    BestPenalty = instance_BestPenalty;
    BetterPenalty = instance_BetterPenalty;
    CurrentPenalty = instance_CurrentPenalty;
    LowerBound = instance_LowerBound;
    
    Runs = instance_Runs;
    Run = instance_Run;
    Trial = instance_Trial;
    MaxTrials = instance_MaxTrials;
    Seed = instance_Seed;
    HashingUsed = instance_HashingUsed;
    HTable = instance_HTable;
    
    TimeLimit = instance_TimeLimit;
    TraceLevel = instance_TraceLevel;
    MergingUsed = instance_MergingUsed;
    MaxCandidates = instance_MaxCandidates;
    BestTour = instance_BestTour;
    Dimension = instance_Dimension;
    CandidateSetSymmetric = instance_CandidateSetSymmetric;
    Excess = instance_Excess;
    ProblemType = instance_ProblemType;
    Rand = instance_Rand;
    Hash = instance_Hash;
}

void LKHSolver::uninstall_globals() {
    // Copy globals back to instance variables after LKH function calls
    copy_globals_to_instance();
}

void LKHSolver::copy_globals_to_instance() {
    extern double StartTime;
    extern int MaxMatrixDimension, CTSPTransform, GTSPSets, DimensionSaved;
    extern struct Node *NodeSet, *FirstNode;
    extern long long MM, BestCost, BetterCost, BestPenalty, BetterPenalty, CurrentPenalty;
    extern int Precision, Norm, Runs, Run, Trial, MaxTrials, HashingUsed;
    extern unsigned int Seed;
    extern HashTable *HTable;
    extern double TimeLimit, LowerBound;
    extern int TraceLevel, MergingUsed, MaxCandidates, Dimension, CandidateSetSymmetric, ProblemType;
    extern int *BestTour;
    extern double Excess;
    extern unsigned *Rand;
    extern unsigned Hash;

    // LKH Global char* filenames that might be changed by LKH
    extern char *ParameterFileName, *ProblemFileName, *TourFileName, *PiFileName, *InitialTourFileName;

    instance_StartTime = StartTime;
    instance_MaxMatrixDimension = MaxMatrixDimension;
    instance_CTSPTransform = CTSPTransform;
    instance_GTSPSets = GTSPSets;
    instance_DimensionSaved = DimensionSaved;
    
    instance_NodeSet = NodeSet;
    instance_FirstNode = FirstNode;
    instance_MM = MM;
    instance_Precision = Precision;
    instance_Norm = Norm;
    
    instance_BestCost = BestCost;
    instance_BetterCost = BetterCost;
    instance_BestPenalty = BestPenalty;
    instance_BetterPenalty = BetterPenalty;
    instance_CurrentPenalty = CurrentPenalty;
    instance_LowerBound = LowerBound;
    
    instance_Runs = Runs;
    instance_Run = Run;
    instance_Trial = Trial;
    instance_MaxTrials = MaxTrials;
    instance_Seed = Seed;
    instance_HashingUsed = HashingUsed;
    instance_HTable = HTable;
    
    instance_TimeLimit = TimeLimit;
    instance_TraceLevel = TraceLevel;
    instance_MergingUsed = MergingUsed;
    instance_MaxCandidates = MaxCandidates;
    instance_BestTour = BestTour;
    instance_Dimension = Dimension;
    instance_CandidateSetSymmetric = CandidateSetSymmetric;
    instance_Excess = Excess;
    instance_ProblemType = ProblemType;
    instance_Rand = Rand;
    instance_Hash = Hash;

    // Helper lambda to update instance string and cstr from LKH global char*
    auto update_instance_filename = [](const char* lkh_global_filename, std::string& instance_path_str, char*& instance_cstr, const char* filename_type) {
        if (lkh_global_filename != nullptr) {
            if (instance_cstr == nullptr || strcmp(lkh_global_filename, instance_cstr) != 0) {
                printf("LKHSolver: Global %s ('%s') differs from instance cstr ('%s'). Updating instance.\n",
                       filename_type, lkh_global_filename, instance_cstr ? instance_cstr : "null");
                instance_path_str = lkh_global_filename;
                delete[] instance_cstr;
                instance_cstr = new char[instance_path_str.length() + 1];
                strcpy(instance_cstr, instance_path_str.c_str());
            }
        } else {
            if (instance_cstr != nullptr) {
                printf("LKHSolver: Global %s is null, but instance cstr ('%s') was set. Clearing instance.\n",
                       filename_type, instance_cstr);
                delete[] instance_cstr;
                instance_cstr = nullptr;
                instance_path_str.clear();
            }
        }
    };

    // Synchronize filename paths that LKH might have modified
    update_instance_filename(ParameterFileName, param_file_path, param_file_cstr, "ParameterFileName");
    update_instance_filename(ProblemFileName, problem_file_path, problem_file_cstr, "ProblemFileName");
    update_instance_filename(TourFileName, tour_file_path, tour_file_cstr, "TourFileName");
    update_instance_filename(PiFileName, pi_file_path, pi_file_cstr, "PiFileName");
    update_instance_filename(InitialTourFileName, initial_tour_file_path, initial_tour_file_cstr, "InitialTourFileName");
}

// File management methods
void LKHSolver::set_parameter_file(const std::string& filename) {
    param_file_path = filename;
    allocate_cstrings();
}

void LKHSolver::set_problem_file(const std::string& filename) {
    problem_file_path = filename;
    allocate_cstrings();
}

void LKHSolver::set_tour_file(const std::string& filename) {
    tour_file_path = filename;
    allocate_cstrings();
}

void LKHSolver::set_pi_file(const std::string& filename) {
    pi_file_path = filename;
    allocate_cstrings();
}

void LKHSolver::set_initial_tour_file(const std::string& filename) {
    initial_tour_file_path = filename;
    allocate_cstrings();
}

// Core functionality implementation
bool LKHSolver::read_parameters() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    try {
        install_globals();
        ReadParameters();
        uninstall_globals();
        initialized = true;
        return true;
    } catch (...) {
        uninstall_globals();
        return false;
    }
}

bool LKHSolver::read_problem() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    try {
        install_globals();
        ReadProblem();
        uninstall_globals();
        problem_loaded = true;
        return true;
    } catch (...) {
        uninstall_globals();
        return false;
    }
}

bool LKHSolver::allocate_structures() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    try {
        install_globals();
        AllocateStructures();
        uninstall_globals();
        structures_allocated = true;
        return true;
    } catch (...) {
        uninstall_globals();
        return false;
    }
}

void LKHSolver::initialize_run_globals(unsigned int seed_val) {
    instance_BestCost = LLONG_MAX;
    instance_BestPenalty = LLONG_MAX;
    instance_CurrentPenalty = LLONG_MAX;
    instance_Runs = 1;
    instance_Run = 1;
    instance_Seed = seed_val;
    
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    install_globals();
    SRandom(seed_val);
    uninstall_globals();
    
    printf("LKHSolver: Run globals initialized. Seed=%u, BestCost=%lld\n", seed_val, instance_BestCost);
}

bool LKHSolver::create_candidate_set() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    try {
        install_globals();
        CreateCandidateSet();
        uninstall_globals();
        return true;
    } catch (...) {
        uninstall_globals();
            return false;
    }
}

void LKHSolver::initialize_statistics() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    install_globals();
    InitializeStatistics();
    uninstall_globals();
}

bool LKHSolver::choose_initial_tour() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    try {
        install_globals();
        ChooseInitialTour();
        uninstall_globals();
        return true;
    } catch (...) {
        uninstall_globals();
        return false;
    }
}

long long LKHSolver::lin_kernighan() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    try {
        install_globals();
        long long result = LinKernighan();
        uninstall_globals();
        return result;
    } catch (...) {
        uninstall_globals();
        return LLONG_MAX;
    }
}

bool LKHSolver::record_better_tour() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    try {
        install_globals();
        RecordBetterTour();
        uninstall_globals();
        return true;
    } catch (...) {
        uninstall_globals();
        return false;
    }
}

bool LKHSolver::record_best_tour() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    try {
        install_globals();
        RecordBestTour();
        uninstall_globals();
        return true;
    } catch (...) {
        uninstall_globals();
            return false;
    }
        }

bool LKHSolver::adjust_candidate_set() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    try {
        install_globals();
        AdjustCandidateSet();
        uninstall_globals();
        return true;
    } catch (...) {
        uninstall_globals();
        return false;
    }
}

bool LKHSolver::prepare_kicking() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    try {
        install_globals();
        PrepareKicking();
        uninstall_globals();
        return true;
    } catch (...) {
        uninstall_globals();
        return false;
    }
}

// State management functions implementation
void LKHSolver::reset_node_tour_fields() {
    if (!instance_FirstNode) {
        printf("ERROR: FirstNode is NULL in reset_node_tour_fields\n");
        return;
    }
    
    Node *t = instance_FirstNode;
    do {
        t->OldPred = t->OldSuc = t->NextBestSuc = t->BestSuc = 0;
    } while ((t = t->Suc) != instance_FirstNode);
    printf("LKHSolver: Node tour fields reset\n");
}

void LKHSolver::select_random_first_node() {
    if (instance_Dimension <= 0) {
        printf("ERROR: Dimension invalid in select_random_first_node\n");
        return;
    }
    
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    install_globals();
    
    extern unsigned Random(void);  // Match the declaration in LKH.h
    if (instance_Dimension == instance_DimensionSaved) {
        instance_FirstNode = &instance_NodeSet[1 + Random() % instance_Dimension];
    } else {
        for (int i = Random() % instance_Dimension; i > 0; i--)
            instance_FirstNode = instance_FirstNode->Suc;
    }
    
    uninstall_globals();
    
    if (instance_FirstNode) 
        printf("LKHSolver: Random FirstNode selected: ID %d\n", instance_FirstNode->Id);
    else 
        printf("ERROR: FirstNode became NULL after selection\n");
}

void LKHSolver::finalize_tour_from_best_suc() {
    if (!instance_FirstNode) {
        printf("ERROR: FirstNode is NULL in finalize_tour_from_best_suc\n");
        return;
    }
    
    printf("LKHSolver: Finalizing tour from BestSuc chain...\n");
    
    Node *t = instance_FirstNode;
    if (instance_Norm == 0 || instance_MaxTrials == 0 || !t->BestSuc) {
        printf("LKHSolver: Setting BestSuc = Suc for all nodes as fallback\n");
        Node* current = instance_FirstNode;
        do {
            if (!current->BestSuc) current->BestSuc = current->Suc;
            current = current->Suc;
        } while (current != instance_FirstNode);
    }
    
    t = instance_FirstNode;
    do {
        if (!t->BestSuc) {
            printf("WARNING: Node %d BestSuc is NULL during finalization. Using t->Suc.\n", t->Id);
            t->BestSuc = t->Suc;
        }
        (t->Suc = t->BestSuc)->Pred = t;
    } while ((t = t->BestSuc) != instance_FirstNode);
    
    if (instance_HashingUsed) {
        instance_Hash = 0;
        t = instance_FirstNode;
        do {
            instance_Hash ^= instance_Rand[t->Id] * instance_Rand[t->Suc->Id];
        } while ((t = t->Suc) != instance_FirstNode);
        printf("LKHSolver: Final Hash recalculated: %u\n", instance_Hash);
    }
}

long long LKHSolver::calculate_penalty() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    install_globals();
    long long p = Penalty();
    uninstall_globals();
    printf("LKHSolver: Penalty calculated: %lld\n", p);
    return p;
}

void LKHSolver::hash_initialize() {
    if (instance_HTable) {
        std::lock_guard<std::mutex> lock(lkh_global_mutex);
        install_globals();
        HashInitialize(instance_HTable);
        uninstall_globals();
        printf("LKHSolver: HashInitialize called\n");
    } else {
        printf("ERROR: HTable is NULL in hash_initialize\n");
    }
}

void LKHSolver::hash_insert(unsigned hash_val, long long cost) {
    if (instance_HTable) {
        std::lock_guard<std::mutex> lock(lkh_global_mutex);
        install_globals();
        HashInsert(instance_HTable, hash_val, cost);
        uninstall_globals();
    } else {
        printf("ERROR: HTable is NULL in hash_insert\n");
    }
}

int LKHSolver::get_first_node_id() const {
    if (instance_FirstNode) return instance_FirstNode->Id;
    return -1;
}

std::vector<int> LKHSolver::get_best_tour() {
    if (instance_BestTour == nullptr || instance_Dimension <= 0) {
        throw std::runtime_error("BestTour is not available or Dimension is invalid");
    }
    
    std::vector<int> tour;
    for (int i = 1; i <= instance_Dimension + 1; i++) {
        tour.push_back(instance_BestTour[i]);
    }
    return tour;
}

bool LKHSolver::validate_solver_state(bool fix_issues) {
    printf("LKHSolver: Validating solver state...\n");
    bool is_valid = true;
    
    if (instance_NodeSet == nullptr) {
        printf("ERROR: NodeSet is null\n");
        is_valid = false;
    } else {
        printf("LKHSolver: NodeSet is valid\n");
    }
    
    if (instance_Dimension <= 0) {
        printf("ERROR: Dimension is invalid (%d)\n", instance_Dimension);
        is_valid = false;
    } else {
        printf("LKHSolver: Dimension is %d\n", instance_Dimension);
    }
    
    if (instance_FirstNode == nullptr) {
        printf("ERROR: FirstNode is null\n");
        if (fix_issues && instance_NodeSet != nullptr && instance_Dimension > 0) {
            printf("LKHSolver: Attempting to initialize FirstNode\n");
            instance_FirstNode = &instance_NodeSet[1];
            
            Node *Prev = instance_FirstNode;
            for (int i = 2; i <= instance_Dimension; i++) {
                Node *N = &instance_NodeSet[i];
                N->Pred = Prev;
                Prev->Suc = N;
                Prev = N;
            }
            Prev->Suc = instance_FirstNode;
            instance_FirstNode->Pred = Prev;
            
            printf("LKHSolver: FirstNode initialized\n");
            is_valid = true;
    } else {
            is_valid = false;
        }
    } else {
        printf("LKHSolver: FirstNode is valid (Id=%d)\n", instance_FirstNode->Id);
    }
    
    printf("LKHSolver: State validation %s\n", is_valid ? "passed" : "failed");
    return is_valid;
}

// Calculate actual tour cost from current tour structure (public, with lock)
long long LKHSolver::calculate_tour_cost() {
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    install_globals();
    try {
        long long cost = calculate_tour_cost_internal();
        uninstall_globals();
        return cost;
    } catch (...) {
        uninstall_globals();
        throw; // Rethrow the exception
    }
}

// High-level solver interface
long long LKHSolver::solve_with_trajectory(int max_trials, double time_limit) {
    // Acquire lock for the entire duration of the solve process
    std::lock_guard<std::mutex> lock(lkh_global_mutex);
    install_globals(); // Install instance state into LKH globals once

    try {

        // Initialize for this run (operates on installed globals)
        // initialize_run_globals calls SRandom, which is global.
        // It also updates instance variables which are then re-installed by install_globals if needed,
        // but here, globals are already installed. So, it directly updates instance_Seed.
        instance_Seed = instance_Seed; // This effectively is already set by constructor or previous run
        SRandom(instance_Seed); // Set LKH seed
        instance_BestCost = LLONG_MAX;     // These are for the current run's tracking within LKH context
        instance_BestPenalty = LLONG_MAX;
        instance_CurrentPenalty = LLONG_MAX;
        instance_Runs = 1;
        instance_Run = 1;
        // Make sure LKH globals reflect this for the run
        BestCost = instance_BestCost;
        BestPenalty = instance_BestPenalty;
        CurrentPenalty = instance_CurrentPenalty;
        Runs = instance_Runs;
        Run = instance_Run;

        // // Initialize and load data if not already done (respecting instance state)
        // if (!read_parameters_internal()) { 
        //     throw std::runtime_error("Failed to read parameters");
        // }
        
        // if (!read_problem_internal()) { 
        //     throw std::runtime_error("Failed to read problem");
        // }
        
        // if (!allocate_structures_internal()) { 
        //     throw std::runtime_error("Failed to allocate structures");
        // }
        
        // Validate state (operates on installed globals)
        if (!validate_solver_state(true)) { // validate_solver_state can remain public as it's read-only mostly
            throw std::runtime_error("Solver state validation failed");
        }

        // Create candidate set (operates on installed globals)
        if (!create_candidate_set_internal()) { 
            throw std::runtime_error("Failed to create candidate set");
        }
        
        // Initialize statistics (operates on installed globals)
        initialize_statistics_internal(); 

        // Validate state (operates on installed globals)
        if (!validate_solver_state(true)) { // validate_solver_state can remain public as it's read-only mostly
            throw std::runtime_error("Solver state validation failed");
        }
        
        // Main solver loop logic (directly calls LKH functions or internal helpers)
        reset_node_tour_fields_internal(); 
        
        instance_BetterCost = LLONG_MAX; 
        instance_BetterPenalty = instance_CurrentPenalty = LLONG_MAX;
        instance_MaxTrials = max_trials; // LKH will use MaxTrials global
        MaxTrials = instance_MaxTrials; // Ensure global is set
        TimeLimit = time_limit; // Set LKH's global TimeLimit
        instance_TimeLimit = time_limit; // And instance
        
        if (instance_MaxTrials > 0) {
            if (instance_HashingUsed) { // HashingUsed is already a global
                hash_initialize_internal(); 
            }
        } else {
            Trial = 1; instance_Trial = 1;
            if (!choose_initial_tour_internal()) { 
                throw std::runtime_error("ChooseInitialTour failed");
            }
            long long current_p = calculate_penalty_internal(); 
            CurrentPenalty = current_p; instance_CurrentPenalty = current_p;
            BetterPenalty = current_p; instance_BetterPenalty = current_p;
        }
        
        if (!prepare_kicking_internal()) { 
            throw std::runtime_error("Failed to prepare kicking");
        }
        
        double solve_start_time = GetTime(); // LKH GetTime
        StartTime = solve_start_time; // LKH global StartTime
        instance_StartTime = solve_start_time; // And instance
        
        for (int trial_count = 1; trial_count <= MaxTrials; trial_count++) {
            if (trial_count > 1 && (GetTime() - StartTime) >= TimeLimit) {
                printf("LKHSolver: Time limit exceeded during trials\n");
                break;
            }
            
            Trial = trial_count; instance_Trial = trial_count;
            
            select_random_first_node_internal(); 
            
            if (!choose_initial_tour_internal()) { 
                printf("LKHSolver: Warning - ChooseInitialTour failed for trial %d\n", trial_count);
                continue;
            }
            
            long long cost_after_lk = lin_kernighan_internal(); 
            
            if (cost_after_lk == LLONG_MAX) {
                printf("LKHSolver: Error in LinKernighan for trial %d\n", trial_count);
                continue;
            }
            
            // LinKernighan updates global CurrentPenalty and global Hash.
            // Cost returned by LinKernighan is already divided by Precision.
            // The instance variables (instance_CurrentPenalty, instance_Hash) will be updated by uninstall_globals.

            bool improved = false;
            if (CurrentPenalty < BetterPenalty || 
                (CurrentPenalty == BetterPenalty && cost_after_lk < BetterCost)) {
                improved = true;
                
                printf("LKHSolver: Trial %d: Improvement! Cost=%lld, Penalty=%lld. (Old BetterCost: %lld, Old BetterPenalty: %lld)\n", 
                       trial_count, cost_after_lk, CurrentPenalty, BetterCost, BetterPenalty);
                
                BetterCost = cost_after_lk;  // RE-ADD: Align with py_lkh_wrapper_base.cpp logic
                BetterPenalty = CurrentPenalty; // RE-ADD: Align with py_lkh_wrapper_base.cpp logic
            
                if (!record_better_tour_internal()) { 
                    printf("LKHSolver: Warning - RecordBetterTour failed in trial %d\n", trial_count);
                }
                
                if (!adjust_candidate_set_internal()) { 
                    printf("LKHSolver: Warning - AdjustCandidateSet failed in trial %d\n", trial_count);
                }
                
                if (!prepare_kicking_internal()) { 
                    printf("LKHSolver: Warning - PrepareKicking failed in trial %d\n", trial_count);
                }
                
                if (instance_HashingUsed) {
                    // HashInitialize might not be needed here again unless specific LKH logic requires it
                    // HashInsert uses the global 'Hash' which LinKernighan would have updated via StoreTour
                    hash_insert_internal(Hash, cost_after_lk); 
                }
            } else {
                 printf("LKHSolver: Trial %d: No improvement. Cost=%lld, Penalty=%lld. (Current BetterCost: %lld, BetterPenalty: %lld)\n", 
                       trial_count, cost_after_lk, CurrentPenalty, BetterCost, BetterPenalty);
            }
        }
        
        finalize_tour_from_best_suc_internal(); 
        
        // if (BetterPenalty != LLONG_MAX) {
        //     CurrentPenalty = BetterPenalty; 
        // }
        
        if (!record_best_tour_internal()) { 
            printf("LKHSolver: Warning - RecordBestTour failed\n");
        }
        
        long long actual_tour_cost = calculate_tour_cost_internal(); 
        printf("LKHSolver: Calculated actual tour cost (from final Suc pointers): %lld\n", actual_tour_cost);
        printf("LKHSolver: LKH global BestCost after run: %lld\n", BetterCost); 
        //printf("LKHSolver: Instance BestCost before uninstall: %lld\n", instance_BestCost);


        uninstall_globals(); 
        // After uninstall, instance_BestCost should be LKH's global BestCost
        printf("LKHSolver: Instance BestCost after uninstall (should match LKH global BestCost): %lld\n", BetterCost);
        
        // The 'actual_tour_cost' is calculated from the Suc pointers of the final tour.
        // LKH's BestCost (now in instance_BestCost) is what LKH considers its best, scaled.
        // We should return the unscaled actual cost of the tour structure that LKH settled on.
        // If penalties are involved, BestCost might be cost+penalty.
        // calculate_tour_cost_internal already divides by precision.
        // Return LKH's recorded BestCost, which should be the most reliable.
        return BetterCost;

    } catch (const std::exception& e) {
        printf("LKHSolver: Exception in solve_with_trajectory: %s\n", e.what());
        uninstall_globals(); 
        throw; 
    } catch (...) {
        printf("LKHSolver: Unknown exception in solve_with_trajectory\n");
        uninstall_globals(); 
        throw; 
    }
}

// ---- Start Internal Helper Methods (without lock/install/uninstall) ----
// These are called by solve_with_trajectory or other public methods that already hold the lock
// and assume LKH globals are already set from the instance.
// They primarily call LKH functions directly.
// Critical: These internal methods DO NOT call install_globals/uninstall_globals themselves.
// They also generally DO NOT update instance variables directly unless it's a simple pointer/value
// that LKH has just set globally and needs to be reflected in the instance for subsequent internal logic
// *before* the final uninstall_globals(). The main state sync happens in the final uninstall_globals().

bool LKHSolver::read_parameters_internal() {
    // Assumes ParameterFileName is set in globals via install_globals()
    ReadParameters(); 
    // LKH's ReadParameters might set other globals like ProblemFileName, MaxTrials etc.
    // These will be copied back to the instance by the outer uninstall_globals().
    // We only need to update the 'initialized' flag for the instance.
    initialized = true; 
    return true;
}

bool LKHSolver::read_problem_internal() {
    // Assumes ProblemFileName is set in globals
    ReadProblem();
    problem_loaded = true;
    return true;
}

bool LKHSolver::allocate_structures_internal() {
    AllocateStructures();
    structures_allocated = true;
    return true;
}

bool LKHSolver::create_candidate_set_internal() {
    CreateCandidateSet();
    return true;
}

void LKHSolver::initialize_statistics_internal() {
    InitializeStatistics();
}

bool LKHSolver::choose_initial_tour_internal() {
    ChooseInitialTour();
    return true;
}

long long LKHSolver::lin_kernighan_internal() {
    return LinKernighan(); // This will update LKH globals (Cost, CurrentPenalty, Hash, tour structure)
}

bool LKHSolver::record_better_tour_internal() {
    RecordBetterTour(); // Updates LKH globals BetterTour, BetterCost, BetterPenalty
    return true;
}

bool LKHSolver::record_best_tour_internal() {
    RecordBestTour(); // Updates LKH globals BestTour, BestCost, BestPenalty
    return true;
}

bool LKHSolver::adjust_candidate_set_internal() {
    AdjustCandidateSet();
    return true;
}

bool LKHSolver::prepare_kicking_internal() {
    PrepareKicking();
    return true;
}

void LKHSolver::reset_node_tour_fields_internal() {
    // Operates on NodeSet pointed to by global FirstNode
    if (!FirstNode) return;
    Node *t = FirstNode;
    do {
        t->OldPred = t->OldSuc = t->NextBestSuc = t->BestSuc = 0;
    } while ((t = t->Suc) != FirstNode);
}

void LKHSolver::select_random_first_node_internal() {
    // Modifies global FirstNode
    if (Dimension <= 0) return;
    extern unsigned Random(void); // LKH's Random
    if (Dimension == DimensionSaved) {
        FirstNode = &NodeSet[1 + Random() % Dimension];
    } else {
        for (int i = Random() % Dimension; i > 0; i--)
            FirstNode = FirstNode->Suc;
    }
}

void LKHSolver::finalize_tour_from_best_suc_internal() {
    // Modifies global FirstNode's Suc/Pred chain and global Hash
    if (!FirstNode) return;
    Node *t = FirstNode;
    if (Norm == 0 || MaxTrials == 0 || !t->BestSuc) {
        do
            t = t->BestSuc = t->Suc;
        while (t != FirstNode);
    }
    //t = FirstNode;
    do
        (t->Suc = t->BestSuc)->Pred = t;
    while ((t = t->BestSuc) != FirstNode);
    
    if (HashingUsed) {
        Hash = 0; 
        t = FirstNode;
        do {
            Hash ^= Rand[t->Id] * Rand[t->Suc->Id];
        } while ((t = t->Suc) != FirstNode);
    }

    if (Trial > MaxTrials)
        Trial = MaxTrials;
    CurrentPenalty = BetterPenalty;
}

long long LKHSolver::calculate_penalty_internal() {
    return Penalty(); // Operates on LKH globals, returns penalty
}

void LKHSolver::hash_initialize_internal() {
    if (HTable) { 
        HashInitialize(HTable); // Operates on global HTable
    }
}

void LKHSolver::hash_insert_internal(unsigned hash_val, long long cost) {
    if (HTable) { 
        HashInitialize(HTable);
        HashInsert(HTable, hash_val, cost); // Operates on global HTable
    }
}

long long LKHSolver::calculate_tour_cost_internal() {
    // Calculates cost from current tour in LKH globals (FirstNode, Suc pointers, C func, Precision)
    if (!FirstNode || Dimension <= 0) { 
        throw std::runtime_error("Tour is not available or Dimension is invalid for calculate_tour_cost_internal");
    }
    long long cost = 0;
    Node *t = FirstNode;
    do {
        cost += C(t, t->Suc); // C is LKH's global cost function
        t = t->Suc;
    } while (t != FirstNode);
    return cost / Precision; // Precision is LKH's global
}

// ---- End Internal Helper Methods ----


// Public methods below will retain their own lock, install_globals, and uninstall_globals
// to ensure they are safe to call independently from Python.

// Initialization methods (public, with lock)
/* Original public methods are now above, calling the _internal ones.
   The definitions from line 1175 to 1398 are redundant and cause redefinition errors.
   Removing them. The existing public methods from lines 510-791 have been updated to correctly
   use the lock, install/uninstall and call the _internal versions.
*/


// Python bindings
PYBIND11_MODULE(lkh_solver, m) {
    m.doc() = "Python bindings for the LKH-AMZ TSP solver with class-based design for multiprocessing";

    py::class_<LKHSolver>(m, "LKHSolver")
        .def(py::init<>())
        
        // File management
        .def("set_parameter_file", &LKHSolver::set_parameter_file)
        .def("set_problem_file", &LKHSolver::set_problem_file)
        .def("set_tour_file", &LKHSolver::set_tour_file)
        .def("set_pi_file", &LKHSolver::set_pi_file)
        .def("set_initial_tour_file", &LKHSolver::set_initial_tour_file)
        
        // Initialization
        .def("read_parameters", &LKHSolver::read_parameters)
        .def("read_problem", &LKHSolver::read_problem)
        .def("allocate_structures", &LKHSolver::allocate_structures)
        .def("initialize_run_globals", &LKHSolver::initialize_run_globals)
        
        // Core solver functions
        .def("create_candidate_set", &LKHSolver::create_candidate_set)
        .def("initialize_statistics", &LKHSolver::initialize_statistics)
        .def("choose_initial_tour", &LKHSolver::choose_initial_tour)
        .def("lin_kernighan", &LKHSolver::lin_kernighan)
        .def("record_better_tour", &LKHSolver::record_better_tour)
        .def("record_best_tour", &LKHSolver::record_best_tour)
        .def("adjust_candidate_set", &LKHSolver::adjust_candidate_set)
        .def("prepare_kicking", &LKHSolver::prepare_kicking)
        
        // State management
        .def("reset_node_tour_fields", &LKHSolver::reset_node_tour_fields)
        .def("select_random_first_node", &LKHSolver::select_random_first_node)
        .def("finalize_tour_from_best_suc", &LKHSolver::finalize_tour_from_best_suc)
        .def("calculate_penalty", &LKHSolver::calculate_penalty)
        
        // Hash functions
        .def("hash_initialize", &LKHSolver::hash_initialize)
        .def("hash_insert", &LKHSolver::hash_insert)
        
        // Getters and setters
        .def("get_best_cost", &LKHSolver::get_best_cost)
        .def("get_better_cost", &LKHSolver::get_better_cost)
        .def("get_better_penalty", &LKHSolver::get_better_penalty)
        .def("get_current_penalty", &LKHSolver::get_current_penalty)
        .def("get_dimension", &LKHSolver::get_dimension)
        .def("get_first_node_id", &LKHSolver::get_first_node_id)
        .def("get_trial_number", &LKHSolver::get_trial_number)
        .def("is_hashing_used", &LKHSolver::is_hashing_used)
        .def("get_lkh_hash", &LKHSolver::get_lkh_hash)
        
        .def("set_better_cost", &LKHSolver::set_better_cost)
        .def("set_better_penalty", &LKHSolver::set_better_penalty)
        .def("set_current_penalty", &LKHSolver::set_current_penalty)
        .def("set_trial_number", &LKHSolver::set_trial_number)
        
        // Tour access
        .def("get_best_tour", &LKHSolver::get_best_tour)
        
        // High-level interface
        .def("solve_with_trajectory", &LKHSolver::solve_with_trajectory, 
             py::arg("max_trials") = 10, py::arg("time_limit") = 3600.0)
        
        // Validation
        .def("validate_solver_state", &LKHSolver::validate_solver_state, 
             py::arg("fix_issues") = true)
        
        // Calculate actual tour cost from current tour structure
        .def("calculate_tour_cost", &LKHSolver::calculate_tour_cost);

    // Simple factory function for creating solver instances (C++11 compatible)
    m.def("create_solver", []() { 
        return new LKHSolver(); 
    }, py::return_value_policy::take_ownership);
}