import math
import random
import multiprocessing
import sys
import os

class Action:
    pass

class AddAction(Action):
    def __init__(self, pos):
        self.pos = pos
    def __eq__(self, other):
        return isinstance(other, AddAction) and self.pos == other.pos
    def __hash__(self):
        return hash(("ADD", self.pos))
    def __repr__(self):
        return f"ADD({self.pos})"

class RemoveAction(Action):
    def __init__(self, pos):
        self.pos = pos
    def __eq__(self, other):
        return isinstance(other, RemoveAction) and self.pos == other.pos
    def __hash__(self):
        return hash(("REMOVE", self.pos))
    def __repr__(self):
        return f"REMOVE({self.pos})"

class MoveAction(Action):
    def __init__(self, pos_from, pos_to):
        self.pos_from = pos_from
        self.pos_to = pos_to
    def __eq__(self, other):
        return isinstance(other, MoveAction) and self.pos_from == other.pos_from and self.pos_to == other.pos_to
    def __hash__(self):
        return hash(("MOVE", self.pos_from, self.pos_to))
    def __repr__(self):
        return f"MOVE({self.pos_from} -> {self.pos_to})"

def distance(p1, p2):
    """Chebyshev distance between grid coordinates"""
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

def get_legal_actions(state, U, k, d_max=None, use_density_masking=False, density_radius=2):
    """
    Action Space Formulation A(s)
    Includes optional Density Masking (Repulsion) to prevent adjacent placements.
    """
    actions = []
    c = len(state)
    
    # Calculate occupied and empty cells
    E = U - state
    
    # Apply Density Masking (Section 5.2)
    # This aggressively prunes the action space during random rollouts
    if c < k:
        for x in E:
            # If masking, skip this cell if it's within density_radius of ANY unit
            if use_density_masking and any(distance(x, u) <= density_radius for u in state):
                continue
            actions.append(AddAction(x))
            
    if c > 0:
        for x in state:
            actions.append(RemoveAction(x))

    if c > 0:
        for x in state:
            for y in E:
                if d_max is None or distance(x, y) <= d_max:
                    # CRITICAL FIX: 'if u != x' tells the mask to ignore the unit that is currently moving!
                    if use_density_masking and any(distance(y, u) <= density_radius for u in state if u != x):
                        continue
                    actions.append(MoveAction(x, y))
                    
    return actions

def apply_action(state, action):
    new_state = set(state)
    if isinstance(action, AddAction):
        new_state.add(action.pos)
    elif isinstance(action, RemoveAction):
        new_state.remove(action.pos)
    elif isinstance(action, MoveAction):
        new_state.remove(action.pos_from)
        new_state.add(action.pos_to)
    return frozenset(new_state)

class MCTSNode:
    """Node Structure holding Canonical State and UCT statistics"""
    def __init__(self, state, legal_actions):
        self.s = frozenset(state)
        self.N = 0
        
        self.legal_actions = legal_actions
        self.N_a = {a: 0 for a in legal_actions}
        self.Q_a = {a: 0.0 for a in legal_actions}
        self.v_L_a = {a: 0 for a in legal_actions}

    def select_action(self, C, q_min, q_max, epsilon=1e-8):
        best_action = None
        best_score = -float('inf')
        
        for a in self.legal_actions:
            n_sa = self.N_a[a]
            v_sa = self.v_L_a[a]
            q_sa = self.Q_a[a]
            
            # --- CRITICAL FIX 1: Q-Normalization Math ---
            # q_sa is the SUM of rewards. Normalizing the sum directly causes huge numerical errors.
            # We must compute the average first, normalize the average, and then factor it.
            if n_sa > 0:
                avg_q = q_sa / n_sa
                if q_max > q_min:
                    norm_avg = (avg_q - q_min) / (q_max - q_min)
                else:
                    norm_avg = 0.5
                norm_sum = norm_avg * n_sa
            else:
                norm_sum = 0.0 # Unvisited nodes get 0 exploitation
                
            denom = n_sa + v_sa + epsilon
            exploitation = norm_sum / denom
            exploration = C * math.sqrt(math.log(max(1, self.N)) / denom)
            
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_action = a
                
        return best_action

def run_mcts(U, k, d_max, H, P, C, iterations, evaluate_batch, initial_state, density_radius, q_min, q_max):
    """
    Main loop with Parallelization via Virtual Loss and Global Transposition Table
    """
    T = {} # Global Transposition Table
    DAG_nodes = {} # Global DAG Registry to prevent State Aliasing
    
    legal_actions = get_legal_actions(initial_state, U, k, d_max, density_radius=density_radius)
    root = MCTSNode(initial_state, legal_actions)
    DAG_nodes[initial_state] = root

    for it in range(iterations):
        paths = []
        rollout_end_states = []
        
        # 4.1 Batched Tree Policy (Selection Phase)
        for p in range(P):
            node = root
            path = []
            visited_in_path = {node.s}
            current_state = node.s
            
            while True:
                if not node.legal_actions:
                    break
                    
                action = node.select_action(C, q_min, q_max)
                path.append((node, action))
                
                # Apply Virtual Loss
                node.v_L_a[action] += 1
                
                current_state = apply_action(node.s, action)
                
                # CYCLE DETECTION: Because actions are reversible (e.g. Move A->B then Move B->A)
                # the selection phase can get stuck in an infinite loop bouncing between DAG nodes.
                if current_state in visited_in_path:
                    break
                    
                visited_in_path.add(current_state)
                
                if current_state not in DAG_nodes:
                    new_legal_actions = get_legal_actions(current_state, U, k, d_max, density_radius=density_radius)
                    child = MCTSNode(current_state, new_legal_actions)
                    DAG_nodes[current_state] = child
                    node = child
                    break
                else:
                    node = DAG_nodes[current_state]
                    
            # 5. Rollout Policy Constraints
            visited = {current_state}
            for _ in range(H):
                actions = get_legal_actions(current_state, U, k, d_max, use_density_masking=True, density_radius=density_radius)

                # Filter out actions that lead to already visited states during this rollout
                candidate_actions = []
                for a in actions:
                    next_s = apply_action(current_state, a)
                    if next_s not in visited:
                        candidate_actions.append(a)

                if not candidate_actions:
                    # No valid moves left in this rollout path
                    break
                    
                # Choose a random action and update the state
                chosen_action = random.choice(candidate_actions)
                # print(f"Chosen Action: {chosen_action}")
                current_state = apply_action(current_state, chosen_action)
                visited.add(current_state)
                
            paths.append(path)
            rollout_end_states.append(current_state)
            
        # 4.2 Batched Evaluation
        max_evals = 3 # Ensure 3 evaluations to smooth out stochastic simulation noise
        states_to_evaluate = []
        
        # 1. Expand the batch to ensure multiple evaluations
        for s in rollout_end_states:
            evals_needed = max_evals
            if s in T:
                evals_needed = max_evals - T[s]['count']
            
            # Add the state multiple times to the batch if needed
            for _ in range(evals_needed):
                states_to_evaluate.append(s)

        if states_to_evaluate:
            rewards = evaluate_batch(states_to_evaluate)
            for s, r in zip(states_to_evaluate, rewards):
                # Update global Q bounds for normalization
                if r < q_min: q_min = r
                if r > q_max: q_max = r

                # Initialize the dictionary if it's the first time seeing this state
                if s not in T:
                    T[s] = {'sum': 0.0, 'count': 0, 'avg': r, 'best_single_run': r}

                # Update the rolling average
                T[s]['sum'] += r
                T[s]['count'] += 1
                T[s]['avg'] = T[s]['sum'] / T[s]['count']

                # Optional: keep track of the absolute best single run for your logs
                if r > T[s]['best_single_run']:
                    T[s]['best_single_run'] = r
                
        # 4.2 Backpropagation with Unique Updates
        
        # We need to make sure we don't over-count N and Q multiple times
        # for the same simulated state in the batch.
        # We aggregate updates per node-action pair
        updates = {}
        for path, s in zip(paths, rollout_end_states):
            reward = T[s]['avg']
            for node, action in path:
                if (node, action) not in updates:
                    updates[(node, action)] = {'reward_sum': 0.0, 'count': 0}
                updates[(node, action)]['reward_sum'] += reward
                updates[(node, action)]['count'] += 1
                
                # We always remove the virtual loss applied earlier
                node.v_L_a[action] -= 1
                
        # Apply the aggregated unique updates
        for (node, action), data in updates.items():
            avg_reward = data['reward_sum'] / data['count']
            node.N_a[action] += 1  # Increment by 1 even if duplicated in batch to prevent over-counting
            node.Q_a[action] += avg_reward
            node.N += 1

        if (it + 1) % 10 == 0 or it == iterations - 1:
            print(
                f"MCTS Iteration {it + 1}/{iterations} | DAG Nodes: {len(DAG_nodes)} | Transposition Table Size: {len(T)}")
            if T:
                best_s = max(T, key=lambda k: T[k]['avg'])
                print(f"Current Best Reward: {T[best_s]['avg']:.2f} | Config (Size {len(best_s)}): {list(best_s)}")

    best_state = max(T, key=lambda k: T[k]['avg'])
    return best_state, T[best_state]['avg'], T