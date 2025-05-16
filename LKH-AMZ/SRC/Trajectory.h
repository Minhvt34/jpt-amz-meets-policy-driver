#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include <stdlib.h> // For malloc, free

// Note: This TrajectoryData struct and its handling are designed to be integrated
// into the LKH C code. The LKH code will be responsible for allocating memory
// (via TD_Initialize) and filling the arrays during its execution.

typedef struct {
    // For each decision step i (0 to TrajectorySize-1):
    
    // --- State representation at step i ---
    int* TourSnapshots;         // Flattened array of tour snapshots.
                                // Each snapshot is of size 'Dimension'.
                                // Access: TourSnapshots[i * Dimension + node_idx_in_tour]
                                // Total size: MaxTrajectorySize * Dimension
    
    int* CurrentNodeAtStep;     // Node ID from which the decision/candidates are considered at step i.
                                // Total size: MaxTrajectorySize

    int* CandidateNodeIds;      // Flattened array of candidate node IDs from CurrentNodeAtStep.
                                // Each step has 'MaxCandidatesPerStep' slots.
                                // Access: CandidateNodeIds[i * MaxCandidatesPerStep + cand_idx]
                                // Total size: MaxTrajectorySize * MaxCandidatesPerStep
                                // Unused candidate slots should be marked with 0 or -1.

    double* CandidateCosts;     // Costs associated with each candidate at step i.
                                // Same layout and size as CandidateNodeIds.
                                
    int* ActualNumCandidatesAtStep; // Actual number of valid candidates recorded for step i.
                                    // (can be less than MaxCandidatesPerStep)
                                    // Total size: MaxTrajectorySize

    long long* TourCostAtStep;  // Cost of the TourSnapshot at step i.
                                // Total size: MaxTrajectorySize

    // --- Action representation for step i ---
    int* ChosenNodeAtStep;      // The node ID chosen from the candidates at step i.
                                // Total size: MaxTrajectorySize

    double* ChosenNodeGainAtStep; // Estimated gain of choosing that node (if LKH calculates this).
                                 // Total size: MaxTrajectorySize

    // --- Control and Metadata ---
    int TrajectorySize;         // Current number of (State, Action) steps recorded.
    int MaxTrajectorySize;      // Maximum capacity for steps (allocated size).
    int Dimension;              // Problem dimension (number of nodes in a full tour).
    int MaxCandidatesPerStep;   // Max candidates stored per step.
    int RecordingEnabled;       // Flag: 1 if recording, 0 otherwise.

} TrajectoryData;

// --- Helper Function Prototypes ---

// Initializes the TrajectoryData structure.
// Allocates memory for all internal arrays.
// problem_dimension: Number of nodes in the TSP problem.
// max_steps: Maximum number of decision steps to record.
// max_candidates_per_step: Max number of candidates to store for each step.
// Returns 0 on success, -1 on failure (e.g., memory allocation).
int TD_Initialize(TrajectoryData* td, int problem_dimension, int max_steps, int max_candidates_per_step);

// Records a state just before a decision is made.
// td: Pointer to the TrajectoryData struct.
// current_node_id: The node from which candidates are being considered.
// current_tour_array: Pointer to an array of 'td->Dimension' integers representing the current tour.
// current_tour_cost: The cost of the current_tour_array.
// lkh_candidate_ids: Array of candidate node IDs from LKH.
// lkh_candidate_costs: Array of costs for lkh_candidate_ids.
// num_lkh_candidates: The number of valid candidates in lkh_candidate_ids/costs.
void TD_RecordState(TrajectoryData* td, 
                    int current_node_id, 
                    const int* current_tour_array, 
                    long long current_tour_cost,
                    const int* lkh_candidate_ids,
                    const double* lkh_candidate_costs,
                    int num_lkh_candidates);

// Records the action taken after a state has been recorded.
// chosen_node_id: The ID of the node selected from the candidates.
// chosen_node_gain: The gain associated with this choice.
// This function assumes a state has just been recorded for the current td->TrajectorySize.
// It then fills in the action part for that same step.
void TD_RecordAction(TrajectoryData* td, int chosen_node_id, double chosen_node_gain);

// Call this to advance to the next step if action is recorded separately or no action for a state.
// Typically, TD_RecordState followed by TD_RecordAction implicitly handles step completion.
// This could be used if a state is recorded but no immediate action follows that fits the model.
// Or if RecordState and RecordAction increment TrajectorySize themselves.
// For now, let TD_RecordAction effectively "complete" the step and increment TrajectorySize.
// So, this function might not be strictly needed if RecordAction increments TrajectorySize.
// Let's assume TD_RecordAction increments TrajectorySize.

// Frees all memory allocated within the TrajectoryData struct.
// Resets sizes and pointers.
void TD_Cleanup(TrajectoryData* td);

// Call this before starting LKH if recording is desired.
void TD_EnableRecording(void);

// Call this if you want to stop recording mid-way.
void TD_DisableRecording(void);

// Records a custom candidate filtering event (if you choose to implement its storage)
void td_record_candidate_filter_event(TrajectoryData* td, int from_node, int to_node, int cost, int accepted);

#endif // TRAJECTORY_H 