#include "Segment.h"
#include "LKH.h"
#include "Hashing.h"
#include "Trajectory.h"

extern TrajectoryData Trajectory;

/*
 * Record a decision point for imitation learning
 * This captures the current state, available candidates, and the chosen move
 */
void RecordDecision(Node *t, Node *selected_candidate_node, long long gain, long long current_tour_cost_from_caller) {
    // Debug print at the very start
    //printff("LKH_C_DEBUG: RecordDecision called. Enabled=%d, Size=%d, MaxSize=%d, Dim=%d, MaxCands=%d\n", 
            // Trajectory.RecordingEnabled, Trajectory.TrajectorySize, Trajectory.MaxTrajectorySize, 
            // Trajectory.Dimension, Trajectory.MaxCandidatesPerStep);

    if (!Trajectory.RecordingEnabled || Trajectory.TrajectorySize >= Trajectory.MaxTrajectorySize) {
        if (Trajectory.RecordingEnabled && Trajectory.TrajectorySize >= Trajectory.MaxTrajectorySize) {
            printff("LKH_C_DEBUG: RecordDecision returning early - trajectory buffer full. Size=%d, MaxSize=%d\n", 
                    Trajectory.TrajectorySize, Trajectory.MaxTrajectorySize);
        }
        return;
    }

    if (Trajectory.Dimension <= 0) { 
        printff("LKH_C_DEBUG: RecordDecision returning early - Trajectory.Dimension is %d\n", Trajectory.Dimension);
        return; 
    }
    int current_tour_array[Trajectory.Dimension];
    Node *current_node_for_tour = FirstNode;
    for (int i = 0; i < Trajectory.Dimension; ++i) {
        if (current_node_for_tour) {
            current_tour_array[i] = current_node_for_tour->Id;
            current_node_for_tour = current_node_for_tour->Suc;
        } else {
            current_tour_array[i] = 0;
        }
    }
    
    long long tour_cost_at_this_state = current_tour_cost_from_caller;

    if (Trajectory.MaxCandidatesPerStep <= 0) { 
        printff("LKH_C_DEBUG: RecordDecision returning early - Trajectory.MaxCandidatesPerStep is %d\n", Trajectory.MaxCandidatesPerStep);
        return; 
    }
    int lkh_candidate_ids[Trajectory.MaxCandidatesPerStep];
    double lkh_candidate_costs[Trajectory.MaxCandidatesPerStep];
    int actual_candidate_count = 0;

    if (t && t->CandidateSet) {
        Candidate *cand_ptr;
        for (cand_ptr = t->CandidateSet; cand_ptr && cand_ptr->To && actual_candidate_count < Trajectory.MaxCandidatesPerStep; cand_ptr++) {
            lkh_candidate_ids[actual_candidate_count] = cand_ptr->To->Id;
            lkh_candidate_costs[actual_candidate_count] = Precision != 0 ? (double)cand_ptr->Cost / Precision : (double)cand_ptr->Cost;
            actual_candidate_count++;
        }
    }

    // Debug print before TD_RecordState
    // printff("LKH_C_DEBUG: Calling TD_RecordState. CurrentNode=%d, ActualCands=%d, TourCost=%lld\n", 
    //         t->Id, actual_candidate_count, tour_cost_at_this_state);

    TD_RecordState(&Trajectory, 
                   t->Id,
                   current_tour_array, 
                   tour_cost_at_this_state,
                   lkh_candidate_ids, 
                   lkh_candidate_costs, 
                   actual_candidate_count);

    int chosen_node_id = selected_candidate_node ? selected_candidate_node->Id : 0;
    double chosen_gain = Precision != 0 ? (double)gain / Precision : (double)gain;

    // Debug print before TD_RecordAction
    // printff("LKH_C_DEBUG: Calling TD_RecordAction. ChosenNode=%d, Gain=%.2f\n", 
    //         chosen_node_id, chosen_gain);

    TD_RecordAction(&Trajectory, chosen_node_id, chosen_gain);
}

/*
 * The LinKernighan function seeks to improve a tour by sequential
 * and non-sequential edge exchanges.
 *
 * The function returns the cost of the resulting tour.
 */

long long LinKernighan()
{
    long long Cost, Gain, G0;
    int X2, i, it = 0;
    Node *t1, *t2, *SUCt1;
    double EntryTime = GetTime();

    Cost = 0;
    Reversed = 0;

    FirstActive = LastActive = 0;
    Swaps = 0;

    /* Compute the cost of the initial tour, Cost.
       Compute the corresponding hash value, Hash.
       Initialize the segment list.
       Make all nodes "active" (so that they can be used as t1). */

    Cost = 0;
    Hash = 0;
    i = 0;
    t1 = FirstNode;
    do {
        t2 = t1->OldSuc = t1->Suc;
        t1->OldPred = t1->Pred;
        t1->Rank = ++i;
        Cost += C(t1, t2) - t1->Pi - t2->Pi;
        if (HashingUsed)
            Hash ^= Rand[t1->Id] * Rand[t2->Id];
        t1->Next = 0;
        if (KickType == 0 || Trial == 1 ||
            !InBestTour(t1, t1->Pred) || !InBestTour(t1, t1->Suc))
            Activate(t1);
    }
    while ((t1 = t1->Suc) != FirstNode);
    Cost /= Precision;
    CurrentPenalty = LLONG_MAX;
    CurrentPenalty = Penalty();
    if (TraceLevel >= 3 ||
        (TraceLevel == 2 &&
         (CurrentPenalty < BetterPenalty ||
          (CurrentPenalty == BetterPenalty && Cost < BetterCost))))
        StatusReport(Cost, EntryTime, "");

    /* Choose t1 as the first "active" node */
    while ((t1 = RemoveFirstActive())) {
        /* t1 is now "passive" */
        SUCt1 = SUC(t1);
        if ((TraceLevel >= 3 || (TraceLevel == 2 && Trial == 1)) &&
            ++it % (Dimension >= 100000 ? 10000 :
                    Dimension >= 10000 ? 1000 : 100) == 0)
            printff("#%d: Time = %0.2f sec.\n",
                    it, fabs(GetTime() - EntryTime));
        /* Choose t2 as one of t1's two neighbors on the tour */
        for (X2 = 1; X2 <= 2; X2++) {
            t2 = X2 == 1 ? PRED(t1) : SUCt1;
            if (Fixed(t1, t2) ||
                (Near(t1, t2) &&
                 (Trial == 1 || KickType == 0)))
                continue;
            G0 = C(t1, t2);
            Swaps = 0;
            PenaltyGain = Gain = 0;

            //printff("LKH_C_LK_DEBUG: About to check Trajectory.RecordingEnabled. Value = %d\n", Trajectory.RecordingEnabled);
            
            if (Trajectory.RecordingEnabled) {
                RecordDecision(t1, t2, G0, Cost); 
            }
                
            /* Try to find a tour-improving move */
            SpecialMove(t1, t2, &G0, &Gain);
            if (PenaltyGain > 0 || Gain > 0) {
                /* An improvement has been found */
                assert(Gain % Precision == 0);
                Cost -= Gain / Precision;
                CurrentPenalty -= PenaltyGain;
                StoreTour();
                if (TraceLevel >= 3 ||
                    (TraceLevel == 2 &&
                     (CurrentPenalty < BetterPenalty ||
                      (CurrentPenalty == BetterPenalty &&
                       Cost < BetterCost))))
                    StatusReport(Cost, EntryTime, "");
                if (HashingUsed && HashSearch(HTable, Hash, Cost))
                    goto End_LinKernighan;
                /* Make t1 "active" again */
                Activate(t1);
                break;
            }
            RestoreTour();
        }
    }
  End_LinKernighan:
    NormalizeNodeList();
    Reversed = 0;
    return Cost;
}
