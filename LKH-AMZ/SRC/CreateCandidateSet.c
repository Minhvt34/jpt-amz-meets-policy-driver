#include "LKH.h"
#include "Trajectory.h"

extern TrajectoryData Trajectory; // Declare Trajectory as an external global

/* 
 * Record candidate filtering decisions
 * This helps understand how LKH selects candidates
 */

void RecordCandidateFiltering(Node *From, Node *To, int Cost, int Accepted) {
    if (!Trajectory.RecordingEnabled || Trajectory.Dimension <= 0) // Added Dimension check
        return;
    
    /* For simplicity, we'll just log to a special section of the trajectory */
    if (Trajectory.TrajectorySize >= Trajectory.MaxTrajectorySize)
        return;
    
    int idx = Trajectory.TrajectorySize;
    
    // Ensure Dimension is valid before using it as an offset
    if (Trajectory.Dimension < 4) { // Need at least 4 slots for this encoding
        // Or handle this error more gracefully, e.g., don't record this type of event
        return; 
    }

    /* Store filter information encoded in the trajectory data structure */
    /* We'll use a special pattern to indicate this is candidate filtering info */
    // This part is highly custom and depends on how you want to interpret this in Python later.
    // The existing TD_RecordState/Action is for tour building steps.
    // This is a different kind of event.
    // For now, let's assume you'll have a separate mechanism or a way to flag these steps.
    // The current TrajectoryData is not really designed for this custom logging.
    //
    // Option 1: Log it as a special state/action if your Python side can differentiate.
    // Option 2: Create a separate logging mechanism for this type of data.
    //
    // Given the current structure, trying to fit it into TourSnapshots is a hack.
    // If you intend to record this, TrajectoryData might need fields for such "meta" events,
    // or you'd call a different recording function.
    //
    // For now, I'll keep your existing logic but highlight it's not standard state/action.
    // It might be better to have a dedicated function in Trajectory.c for this if it's frequent.

    td_record_candidate_filter_event(&Trajectory, From->Id, To->Id, Cost, Accepted);


    // Trajectory.TourSnapshots[idx * Trajectory.Dimension] = -1;  /* Special marker for filter event */
    // Trajectory.TourSnapshots[idx * Trajectory.Dimension + 1] = From->Id;
    // Trajectory.TourSnapshots[idx * Trajectory.Dimension + 2] = To->Id;
    // Trajectory.TourSnapshots[idx * Trajectory.Dimension + 3] = Accepted; // 1 for accepted, 0 for rejected
    
    // Trajectory.CandidateCosts[idx * Trajectory.MaxCandidatesPerStep] = Cost; // Storing cost in the first slot
    // Trajectory.ActualNumCandidatesAtStep[idx] = 0; // No "candidates" in the usual sense for this event
    // Trajectory.CurrentNodeAtStep[idx] = From->Id;
    // Trajectory.ChosenNodeAtStep[idx] = To->Id; // Node being considered
    // Trajectory.ChosenNodeGainAtStep[idx] = Accepted; // Re-using gain for accepted status

    // Trajectory.TrajectorySize++; // Increment size for this special event
}

// A new function prototype that should go into Trajectory.h
// And implemented in Trajectory.c
// void td_record_candidate_filter_event(TrajectoryData* td, int from_node, int to_node, int cost, int accepted);


/*
 * The CreateCandidateSet function determines for each node its set of incident
 * candidate edges.
 *
 * The Ascent function is called to determine a lower bound on the optimal tour 
 * using subgradient optimization. But only if the penalties (the Pi-values) is
 * not available on file. In the latter case, the penalties is read from the 
 * file, and the lower bound is computed from a minimum 1-tree.      
 *
 * The function GenerateCandidates is called to compute the Alpha-values and to 
 * associate to each node a set of incident candidate edges.  
 *
 * The CreateCandidateSet function itself is called from LKHmain.
 */

void CreateCandidateSet()
{
    long long Cost, MaxAlpha;
    Node *Na;
    int i;
    double EntryTime = GetTime();

    Norm = 9999;
    if (C == C_EXPLICIT) {
        Na = FirstNode;
        do {
            for (i = 1; i < Na->Id; i++)
                Na->C[i] *= Precision;
        }
        while ((Na = Na->Suc) != FirstNode);
    }
    if (TraceLevel >= 2)
        printff("Creating candidates ...\n");
    Na = FirstNode;
    do
        Na->Pi = 0;
    while ((Na = Na->Suc) != FirstNode);
    Cost = Ascent();
    if (MaxCandidates > 0) {
        if (TraceLevel >= 2)
            printff("Computing lower bound ... ");
        Cost = Minimum1TreeCost(0);
        if (TraceLevel >= 2)
            printff("done\n");
    } else {
        if (TraceLevel >= 2)
            printff("Computing lower bound ... ");
        Cost = Minimum1TreeCost(1);
        if (TraceLevel >= 2)
            printff("done\n");
    }
    LowerBound = (double) Cost / Precision;
    if (TraceLevel >= 1) {
        printff("Lower bound = %0.1f", LowerBound);
        printff(", Ascent time = %0.2f sec.",
                fabs(GetTime() - EntryTime));
        printff("\n");
    }
    MaxAlpha = (long long) fabs(Excess * Cost);
    GenerateCandidates(MaxCandidates, MaxAlpha, CandidateSetSymmetric);

    if (MaxTrials > 0) {
        Na = FirstNode;
        do {
            if (!Na->CandidateSet || !Na->CandidateSet[0].To) {
                if (MaxCandidates == 0)
                    eprintf
                        ("MAX_CANDIDATES = 0: Node %d has no candidates",
                         Na->Id);
                else
                    eprintf("Node %d has no candidates", Na->Id);
            }
        }
        while ((Na = Na->Suc) != FirstNode);
    }
    if (C == C_EXPLICIT) {
        Na = FirstNode;
        do
            for (i = 1; i < Na->Id; i++)
                Na->C[i] += Na->Pi + NodeSet[i].Pi;
        while ((Na = Na->Suc) != FirstNode);
    }
    if (TraceLevel >= 1) {
        CandidateReport();
        printff("Preprocessing time = %0.2f sec.\n",
                fabs(GetTime() - EntryTime));
    }
}
