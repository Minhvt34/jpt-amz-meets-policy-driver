#include "Trajectory.h"
#include "LKH.h" // For LKH specific types if needed, and for printff etc.
#include <stdio.h>  // For printf in error messages
#include <string.h> // For memcpy, memset


// Global instance of TrajectoryData. This will be accessed by LKH C code
// and by the py_lkh_wrapper.cpp (via extern declaration).
TrajectoryData Trajectory; 

int TD_Initialize(TrajectoryData* td, int problem_dimension, int max_steps, int max_candidates_per_step) {
    if (td == NULL || problem_dimension <= 0 || max_steps <= 0 || max_candidates_per_step <= 0) {
        // Consider using LKH's eprintf or printff for errors
        fprintf(stderr, "TD_Initialize: Invalid parameters.\n");
        return -1;
    }

    td->TrajectorySize = 0;
    td->Dimension = problem_dimension;
    td->MaxTrajectorySize = max_steps;
    td->MaxCandidatesPerStep = max_candidates_per_step;
    td->RecordingEnabled = 0; // Disabled by default, enable with TD_EnableRecording

    // Allocate memory for all arrays
    // Size = MaxTrajectorySize * Dimension * sizeof(int)
    td->TourSnapshots = (int*)malloc(max_steps * problem_dimension * sizeof(int));
    // Size = MaxTrajectorySize * sizeof(int)
    td->CurrentNodeAtStep = (int*)malloc(max_steps * sizeof(int));
    // Size = MaxTrajectorySize * MaxCandidatesPerStep * sizeof(int)
    td->CandidateNodeIds = (int*)malloc(max_steps * max_candidates_per_step * sizeof(int));
    // Size = MaxTrajectorySize * MaxCandidatesPerStep * sizeof(double)
    td->CandidateCosts = (double*)malloc(max_steps * max_candidates_per_step * sizeof(double));
    // Size = MaxTrajectorySize * sizeof(int)
    td->ActualNumCandidatesAtStep = (int*)malloc(max_steps * sizeof(int));
    // Size = MaxTrajectorySize * sizeof(long long)
    td->TourCostAtStep = (long long*)malloc(max_steps * sizeof(long long));
    // Size = MaxTrajectorySize * sizeof(int)
    td->ChosenNodeAtStep = (int*)malloc(max_steps * sizeof(int));
    // Size = MaxTrajectorySize * sizeof(double)
    td->ChosenNodeGainAtStep = (double*)malloc(max_steps * sizeof(double));

    // Check for allocation failures
    if (!td->TourSnapshots || !td->CurrentNodeAtStep || !td->CandidateNodeIds || 
        !td->CandidateCosts || !td->ActualNumCandidatesAtStep || !td->TourCostAtStep ||
        !td->ChosenNodeAtStep || !td->ChosenNodeGainAtStep) {
        fprintf(stderr, "TD_Initialize: Memory allocation failed.\n");
        TD_Cleanup(td); // Free any partially allocated memory
        return -1;
    }
    
    // Initialize with zeros or sentinels if desired (optional, as LKH will fill)
    memset(td->CandidateNodeIds, 0, max_steps * max_candidates_per_step * sizeof(int));

    return 0; 
}

void TD_RecordState(
    TrajectoryData* td, 
    int current_node_id, 
    const int* current_tour_array, // Assumed to be of size td->Dimension
    long long current_tour_cost,
    const int* lkh_candidate_ids,      // Array from LKH
    const double* lkh_candidate_costs,  // Array from LKH
    int num_lkh_candidates) {

    if (!td || !td->RecordingEnabled || td->TrajectorySize >= td->MaxTrajectorySize) {
        return; // Not initialized, not recording, or buffer full
    }

    int step_idx = td->TrajectorySize; // Current step we are about to record

    // 1. Record Tour Snapshot
    if (current_tour_array) {
        memcpy(&td->TourSnapshots[step_idx * td->Dimension],
               current_tour_array,
               td->Dimension * sizeof(int));
    }

    // 2. Record Current Node ID
    td->CurrentNodeAtStep[step_idx] = current_node_id;

    // 3. Record Tour Cost
    td->TourCostAtStep[step_idx] = current_tour_cost;

    // 4. Record Candidates
    int candidates_to_copy = num_lkh_candidates;
    if (candidates_to_copy > td->MaxCandidatesPerStep) {
        candidates_to_copy = td->MaxCandidatesPerStep;
        // Optionally log that candidates were truncated
    }
    td->ActualNumCandidatesAtStep[step_idx] = candidates_to_copy;

    if (lkh_candidate_ids) {
        memcpy(&td->CandidateNodeIds[step_idx * td->MaxCandidatesPerStep],
               lkh_candidate_ids,
               candidates_to_copy * sizeof(int));
    }
    if (lkh_candidate_costs) {
        memcpy(&td->CandidateCosts[step_idx * td->MaxCandidatesPerStep],
               lkh_candidate_costs,
               candidates_to_copy * sizeof(double));
    }
    
    // Fill remaining candidate slots for this step with a sentinel (e.g., 0 or -1)
    for (int k = candidates_to_copy; k < td->MaxCandidatesPerStep; ++k) {
        td->CandidateNodeIds[step_idx * td->MaxCandidatesPerStep + k] = 0; // Sentinel
        td->CandidateCosts[step_idx * td->MaxCandidatesPerStep + k] = 0.0; // Sentinel
    }
    
    // Note: TrajectorySize is NOT incremented here. 
    // It's incremented by TD_RecordAction, effectively completing the (State, Action) pair.
}

void TD_RecordAction(TrajectoryData* td, int chosen_node_id, double chosen_node_gain) {
    if (!td || !td->RecordingEnabled || td->TrajectorySize >= td->MaxTrajectorySize) {
        return; // Not initialized, not recording, or buffer full
    }

    int step_idx = td->TrajectorySize; // Current step (for which state was just recorded)

    td->ChosenNodeAtStep[step_idx] = chosen_node_id;
    td->ChosenNodeGainAtStep[step_idx] = chosen_node_gain;

    // This action completes the current (State, Action) pair, so advance.
    td->TrajectorySize++;
}

void TD_Cleanup(TrajectoryData* td) {
    if (td == NULL) return;

    free(td->TourSnapshots);
    free(td->CurrentNodeAtStep);
    free(td->CandidateNodeIds);
    free(td->CandidateCosts);
    free(td->ActualNumCandidatesAtStep);
    free(td->TourCostAtStep);
    free(td->ChosenNodeAtStep);
    free(td->ChosenNodeGainAtStep);

    // Reset fields to indicate an uninitialized state
    td->TourSnapshots = NULL;
    td->CurrentNodeAtStep = NULL;
    td->CandidateNodeIds = NULL;
    td->CandidateCosts = NULL;
    td->ActualNumCandidatesAtStep = NULL;
    td->TourCostAtStep = NULL;
    td->ChosenNodeAtStep = NULL;
    td->ChosenNodeGainAtStep = NULL;
    
    td->TrajectorySize = 0;
    td->MaxTrajectorySize = 0;
    td->Dimension = 0;
    td->MaxCandidatesPerStep = 0;
    td->RecordingEnabled = 0;
}

void TD_EnableRecording() {
    // Ensure Trajectory is initialized before enabling recording
    if (Trajectory.Dimension == 0 || Trajectory.MaxTrajectorySize == 0) {
        printf("LKH_C_DEBUG: TD_EnableRecording: ERROR - Trajectory not properly initialized (Dimension: %d, MaxSize: %d). Cannot enable recording.\\n", Trajectory.Dimension, Trajectory.MaxTrajectorySize);
        Trajectory.RecordingEnabled = 0; // Explicitly ensure it's off
        return;
    }
    Trajectory.RecordingEnabled = 1;
    // Trajectory.TrajectorySize = 0; // Resetting size here might be premature if multiple segments are recorded or if initialization handles it.
                                     // Let's assume TD_Initialize sets TrajectorySize to 0.
    printf("LKH_C_DEBUG: TD_EnableRecording: Recording ENABLED. Trajectory.Dimension = %d\\n", Trajectory.Dimension);
}

void TD_DisableRecording() {
    Trajectory.RecordingEnabled = 0;
    printf("LKH_C_DEBUG: TD_DisableRecording: Recording DISABLED.\\n");
}

// Implementation for the custom candidate filtering event recorder
void td_record_candidate_filter_event(TrajectoryData* td, int from_node, int to_node, int cost, int accepted) {
    // This is a placeholder implementation.
    // You need to decide if and how you want to store this specific type of event.
    // For now, it does nothing other than potentially printing, which will satisfy the linker.
    // To actually store this, you would add new fields to TrajectoryData struct
    // and manage them similarly to how TD_RecordState/Action work (e.g., incrementing a
    // separate counter or using a portion of the main TrajectorySize if carefully managed).
    if (td && td->RecordingEnabled && TraceLevel >= 3) { // Example: only print if TraceLevel is high
        printff("LKH_TRAJ_DEBUG: Candidate filter event: From %d, To %d, Cost %d, Accepted %d (Current TrajSize: %d)\n", 
                from_node, to_node, cost, accepted, td->TrajectorySize);
    }
    // IMPORTANT: If you want this event to consume a slot in the main trajectory arrays 
    // and increment td->TrajectorySize, you must implement that logic here carefully.
    // The current RecordCandidateFiltering function in CreateCandidateSet.c implies it *would* 
    // consume a slot if its commented-out code were active.
    // If so, this function should take over that responsibility.
    // For now, this stub does NOT increment TrajectorySize or use the main arrays.
} 