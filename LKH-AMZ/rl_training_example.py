import os
import sys
import numpy as np
import random
from collections import deque

# Add the SRC directory to find the Python module
module_path = os.path.join(os.path.dirname(__file__), 'SRC')
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    import lkh_solver
    print(f"lkh_solver module loaded from: {lkh_solver.__file__}")
except ImportError as e:
    print(f"Error importing lkh_solver: {e}")
    print("Please ensure the module is built with the reinforcement learning extensions.")
    sys.exit(1)

# Simple neural network for policy learning
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    
    # Only define the network class if PyTorch is available
    class SimpleNetwork(nn.Module):
        def __init__(self, input_size, output_size):
            super(SimpleNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, output_size)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
except ImportError:
    print("PyTorch not available. Running in demonstration mode only.")

# Constants
MAX_BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001
NUM_EPISODES = 100
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        return len(self.buffer)

def state_to_features(state):
    """Convert LKH state to feature vector for neural network input"""
    # Simple feature: current position and one-hot encoding of candidates
    features = []
    
    # Position
    position_feature = np.zeros(state.dimension if hasattr(state, 'dimension') and state.dimension > 0 else 100)  # Assuming max 100 nodes if not specified
    if state.current_node -1 < len(position_feature): # Check bounds
        position_feature[state.current_node - 1] = 1
    features.append(position_feature)
    
    # Candidate nodes
    candidates_feature = np.zeros(state.dimension if hasattr(state, 'dimension') and state.dimension > 0 else 100)
    for node in state.candidate_nodes:
        if node - 1 < len(candidates_feature): # Check bounds
            candidates_feature[node - 1] = 1
    features.append(candidates_feature)
    
    # Candidate costs (normalized)
    if state.candidate_costs:
        costs = np.array(state.candidate_costs)
        max_cost = max(costs)
        if max_cost > 0:
            normalized_costs = costs / max_cost
        else:
            normalized_costs = costs
    else:
        normalized_costs = np.array([])
    
    # Pad to fixed length
    padded_costs = np.zeros(state.dimension if hasattr(state, 'dimension') and state.dimension > 0 else 100)
    
    # Ensure normalized_costs is not empty and its length does not exceed padded_costs
    if normalized_costs.size > 0:
        num_elements_to_copy = min(len(normalized_costs), len(padded_costs))
        padded_costs[:num_elements_to_copy] = normalized_costs[:num_elements_to_copy]
        
    features.append(padded_costs)
    
    # Current tour length
    tour_length = len(state.current_tour)
    tour_feature = np.zeros(1)
    tour_feature[0] = tour_length / 100.0  # Normalize
    features.append(tour_feature)
    
    return np.concatenate(features)

def collect_expert_trajectories(problem_files, param_files, num_trajectories=1):
    """Collect expert trajectories from LKH for imitation learning"""
    trajectories = []
    
    # Define a maximum number of steps for a single trajectory
    # This should be large enough to capture significant portions of LKH's search
    max_steps_per_trajectory = 2000 # Example value, can be tuned

    for _ in range(num_trajectories):
        problem_idx = random.randint(0, len(problem_files) - 1)
        problem_file = problem_files[problem_idx]
        param_file = param_files[problem_idx]
        
        print(f"Collecting expert trajectory using {problem_file} with param {param_file}, max_steps={max_steps_per_trajectory}")
        # Pass the new max_trajectory_steps argument
        trajectory = lkh_solver.solve_and_record_trajectory(problem_file, param_file, max_steps_per_trajectory)
        trajectories.append(trajectory)
        
        print(f"Collected trajectory with {len(trajectory.states)} states (recorded steps: {trajectory.recorded_steps}).")
        print(f"  Trajectory details: Dimension={trajectory.dimension}, MaxCandidatesPerStep={trajectory.max_candidates_per_step}, FinalCost={trajectory.final_cost}")
        if trajectory.recorded_steps > 0 and trajectory.states:
            print(f"  First state sample: current_node={trajectory.states[0].current_node}, tour_cost={trajectory.states[0].tour_cost}, num_actual_cands={trajectory.states[0].actual_num_candidates}")
            if trajectory.actions:
                 print(f"  First action sample: chosen_node={trajectory.actions[0].chosen_node}, gain={trajectory.actions[0].gain}")

    return trajectories

def pretrain_with_imitation(model, trajectories):
    """Pretrain policy model using expert demonstrations from LKH"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping pretraining.")
        return
    
    print("Pretraining policy with imitation learning...")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):  # 10 epochs of pretraining
        total_loss = 0
        examples = 0
        
        for trajectory in trajectories:
            for i in range(len(trajectory.states)):
                state = trajectory.states[i]
                action = trajectory.actions[i]
                
                state_tensor = torch.FloatTensor(state_to_features(state)).unsqueeze(0)
                
                # Get model prediction
                action_probs = model(state_tensor)
                
                # Prepare target
                target_action = torch.LongTensor([action.chosen_node - 1])
                
                # Compute loss
                loss = criterion(action_probs, target_action)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                examples += 1
        
        avg_loss = total_loss / max(1, examples)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
    
    print("Pretraining complete.")

def train_reinforcement(model, problem_files):
    """Train policy model with reinforcement learning after imitation pretraining"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping RL training.")
        return
    
    print("Starting reinforcement learning training...")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(MAX_BUFFER_SIZE)
    epsilon = EPSILON_START
    
    for episode in range(NUM_EPISODES):
        # Choose a random problem for this episode
        problem_idx = random.randint(0, len(problem_files) - 1)
        problem_file = problem_files[problem_idx]
        param_file = param_files[problem_idx] # Get corresponding param file

        print(f"--- Episode {episode+1}: Using problem {problem_file} and param {param_file} ---")

        # Initialize LKH environment using granular functions
        lkh_solver.set_problem_file(problem_file)
        lkh_solver.set_parameter_file(param_file)
        lkh_solver.LKH_ReadParameters()
        lkh_solver.LKH_ReadProblem()
        current_dimension = lkh_solver.get_dimension()
        if current_dimension <= 0:
            print(f"Error: LKH Dimension is {current_dimension} after loading problem. Skipping episode.")
            continue
        lkh_solver.LKH_AllocateStructures()
        lkh_solver.LKH_CreateCandidateSet() # Crucial for getting candidate lists
        lkh_solver.LKH_InitializeStatistics()
        lkh_solver.set_seed_wrapper(random.randint(0, 65535)) # Set a random seed for this run

        # Manually construct initial state
        state = lkh_solver.LKHState_py()
        state.dimension = current_dimension
        state.current_node = 1 # Typically start at node 1
        state.current_tour = [1]
        state.tour_cost = 0
        # TODO: Populate initial candidates for node 1.
        # This requires a new C++ helper: get_candidates_for_node(node_id)
        # For now, a simplified version:
        state.candidate_nodes = [n for n in range(2, min(state.dimension + 1, 11))] # Dummy: first few other nodes
        state.candidate_costs = [100.0] * len(state.candidate_nodes) # Dummy costs
        state.actual_num_candidates = len(state.candidate_nodes)
        
        done = False
        total_reward = 0
        
        while not done:
            # Convert state to feature vector
            state_features = state_to_features(state)
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Random action
                if state.candidate_nodes:
                    action_idx = random.randint(0, len(state.candidate_nodes) - 1)
                    action_node = state.candidate_nodes[action_idx]
                else:
                    # No valid moves, end episode
                    done = True
                    continue
            else:
                # Model prediction
                with torch.no_grad():
                    q_values = model(state_tensor).detach().numpy()[0]
                
                # Filter to only valid actions
                valid_q = {node-1: q_values[node-1] for node in state.candidate_nodes}
                if valid_q:
                    action_node = max(valid_q.items(), key=lambda x: x[1])[0] + 1
                else:
                    # No valid moves, end episode
                    done = True
                    continue
            
            # Create action object
            action = lkh_solver.LKHAction_py()
            action.chosen_node = action_node # Corrected from selected_node
            # action.gain is not set by Python agent, comes from LKH trajectory
            
            # Take action in environment
            # NOTE: lkh_solver.step_environment is still the C++ stub.
            # The Python logic below simulates the step.
            next_state = lkh_solver.step_environment(state, action) # This call uses the C++ stub
            
            # --- Python-side simulation of step based on 'action.chosen_node' ---
            # This part would ideally be replaced by a richer C++ step_environment if it could
            # update LKH's state and provide the true next state and candidates.
            
            # For now, we manually update a Python-managed representation of the next state:
            py_next_state = lkh_solver.LKHState_py()
            py_next_state.dimension = state.dimension
            py_next_state.current_tour = state.current_tour + [action.chosen_node]
            py_next_state.current_node = action.chosen_node
            py_next_state.tour_cost = state.tour_cost + 1 # Placeholder cost increment

            # Regenerate candidates (simplistic: all unvisited nodes up to a limit)
            temp_candidates = []
            visited_nodes = set(py_next_state.current_tour)
            for i in range(1, py_next_state.dimension + 1):
                if i not in visited_nodes:
                    temp_candidates.append(i)
                if len(temp_candidates) >= 10: # Limit for dummy candidates
                    break
            py_next_state.candidate_nodes = temp_candidates
            py_next_state.candidate_costs = [100.0] * len(py_next_state.candidate_nodes) # Dummy costs
            py_next_state.actual_num_candidates = len(py_next_state.candidate_nodes)
            # --- End of Python-side simulation of step ---

            # Calculate reward - negative distance to encourage shortest path
            reward = -1.0  # Simple reward: penalize each step
            
            # Check if tour is complete (no more candidates according to Python sim)
            if not py_next_state.candidate_nodes or len(py_next_state.current_tour) == py_next_state.dimension:
                done = True
                
                # Create complete tour (ensure it's closed)
                if py_next_state.current_tour[0] != py_next_state.current_tour[-1] and py_next_state.dimension > 1:
                     complete_tour = py_next_state.current_tour + [py_next_state.current_tour[0]]
                else:
                     complete_tour = py_next_state.current_tour
                
                # Ensure tour is not empty and has valid structure for evaluation
                if not complete_tour or len(complete_tour) < 2 : # Basic check
                    print(f"Episode {episode+1}, Invalid or too short tour: {complete_tour}. Assigning high penalty cost.")
                    final_cost = float(state.dimension * 10000) # Penalize heavily
                else:
                    print(f"Episode {episode+1}, Evaluating tour: {complete_tour} for problem: {problem_file}")
                    final_cost = lkh_solver.evaluate_solution(problem_file, complete_tour)
                    print(f"Episode {episode+1}, Tour Cost from LKH: {final_cost:.1f}")
                
                # Give terminal reward based on tour quality
                reward = -final_cost / 1000.0  # Scale down
            
            # Store in replay buffer
            buffer.add(state_features, action_node, reward, state_to_features(next_state), done)
            
            # Update state
            state = py_next_state # Use the Python-simulated next state
            total_reward += reward
            
            # Learn from experience
            if len(buffer) > BATCH_SIZE:
                # Sample mini-batch
                batch = buffer.sample(BATCH_SIZE)
                
                # Unpack batch
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*batch)
                
                # Convert to tensors
                states_tensor = torch.FloatTensor(np.array(states_batch))
                actions_tensor = torch.LongTensor(np.array([a-1 for a in actions_batch]))  # Adjust indexing
                rewards_tensor = torch.FloatTensor(np.array(rewards_batch))
                next_states_tensor = torch.FloatTensor(np.array(next_states_batch))
                dones_tensor = torch.FloatTensor(np.array(dones_batch))
                
                # Get current Q values
                current_q = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                
                # Get next Q values
                next_q = torch.zeros(len(batch))
                with torch.no_grad():
                    next_q = model(next_states_tensor).max(1)[0]
                
                # Compute target Q values
                target_q = rewards_tensor + GAMMA * next_q * (1 - dones_tensor)
                
                # Compute loss
                loss = nn.MSELoss()(current_q, target_q)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        print(f"Episode {episode+1}, Total Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")
    
    print("Reinforcement learning training complete.")

def main():
    # Define problem and parameter files
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data-evaluation")
    
    # Look for .tsp problem files
    problem_files = []
    param_files = []
    
    # Example path pattern - adjust to match your data
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".tsp"):
                problem_file = os.path.join(root, file)
                problem_files.append(problem_file)
                
                # Assuming .par file has same base name
                base_name = os.path.splitext(file)[0]
                param_file_path = os.path.join(root, f"{base_name}.par")
                if os.path.exists(param_file_path):
                    param_files.append(param_file_path)
                else:
                    # Use default parameter file if specific one not exists
                    param_file_path = os.path.join(os.path.dirname(__file__), "default.par")
                    param_files.append(param_file_path) # Use common default
    
    if not problem_files:
        print("No problem files found in data_dir. Using a specific fallback problem file for demonstration...")
        
        # Use the user-specified CTSPTW problem file
        specific_problem_file = "/home/minhvt/jpt-amz/data-evaluation/model_apply_outputs/TSPLIB_1/amz0000.ctsptw"
        # Use the corresponding specific parameter file
        specific_param_file = "/home/minhvt/jpt-amz/data-evaluation/model_apply_outputs/TSPLIB_1/amz0000.par"

        # Check if the specific problem and parameter files exist
        if not os.path.exists(specific_problem_file):
            print(f"Error: Specified problem file {specific_problem_file} does not exist!")
            sys.exit(1)
        if not os.path.exists(specific_param_file):
            print(f"Error: Specified parameter file {specific_param_file} does not exist!")
            sys.exit(1)
            
        problem_files = [specific_problem_file]
        param_files = [specific_param_file]
    
    print(f"Found {len(problem_files)} problem files.")
    
    # Set up neural network model
    model = None
    if TORCH_AVAILABLE:
        input_size = 301  # position(100) + candidates(100) + costs(100) + tour_length(1)
        output_size = 100  # Assuming max 100 nodes in problems
        model = SimpleNetwork(input_size, output_size)
    
    # Collect expert trajectories from LKH
    print("Collecting expert trajectories...")
    # Let's collect only 1 trajectory for now to simplify debugging output
    trajectories = collect_expert_trajectories(problem_files, param_files, num_trajectories=1)
    
    # Pretrain with imitation learning
    if model is not None:
        pretrain_with_imitation(model, trajectories)
        
        # Further train with reinforcement learning
        train_reinforcement(model, problem_files)
        
        # Save the trained model
        torch.save(model.state_dict(), "lkh_rl_model.pt")
        print("Model saved to lkh_rl_model.pt")
    else:
        print("\nDemonstration Complete: Successfully collected expert trajectories from LKH.")
        print("Install PyTorch to enable neural network training and reinforcement learning.")

if __name__ == "__main__":
    main() 