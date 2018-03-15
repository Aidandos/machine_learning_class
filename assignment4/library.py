#
# USI - Universita della svizzera italiana
#
# Machine Learning - Fall 2017
#
# Assignment 3: Reinforcement Learning
#
# (!) Please do NOT write code outside of functions in this file.
# (!) Do not change the name or parameters of the functions that you have to implement.
# (!) The functions below will be run and checked by a script using a different environment.
#

import numpy as np

E = EMPTY = 0
B = BLOCKED = 1
G = GOAL = 2

# The maze from the assignment
MAZE = np.array(
    [[0, 0, 0, 0, 0, 0, 0],
     [0, 0, B, 0, 0, 0, 0],
     [0, B, B, 0, 0, 0, 0],
     [0, 0, B, 0, 0, B, 0],
     [0, 0, 0, 0, 0, B, G]])

# The possible actions
ACTIONS = np.array(['N', 'E', 'S', 'W'])
ACTION_IDX = {a: idx for idx, a in enumerate(ACTIONS)}

# transition probabilities given a specific action to one of the 5 outcomes.
# probabilties for [staying, N, E, S, W]
ACTION_PROBS = {
    ACTIONS[0]: [0.0, 0.7, 0.1, 0.1, 0.1],
    ACTIONS[1]: [0.0, 0.1, 0.7, 0.1, 0.1],
    ACTIONS[2]: [0.0, 0.1, 0.1, 0.7, 0.1],
    ACTIONS[3]: [0.0, 0.1, 0.1, 0.1, 0.7]
}


def get_neighbours(state):
    """ Return a list of all four neighbour states and the current position. """
    row, col = state
    return [
        (row,   col),    # -
        (row - 1, col),    # N
        (row,   col + 1),  # E
        (row + 1, col),    # S
        (row,   col - 1),  # W
    ]


def is_in_maze(state):
    """ Return True if position is inside the maze and not blocked. """
    R, C = MAZE.shape
    row, col = state
    return (0 <= row < R) and (0 <= col < C) and MAZE[state] != BLOCKED


def state_prob(s, s2, a):
    """ Return the probability of transitioning from s to s2 by action a. """
    assert is_in_maze(s)
    assert a in ACTIONS
    neighbours = get_neighbours(s)
    assert s2 in neighbours

    if MAZE[s] == GOAL:
        return 1.0 if s == s2 else 0.0

    # copy action a transition probabilities
    pr = list(ACTION_PROBS[a])

    # fall back to s if you can't go to a specific neighbour.
    for i, n in enumerate(neighbours):
        if not is_in_maze(n):
                # illegal neighbour, move probabilities to staying.
            pr[0] += pr[i]
            # reset the probability of the illegal move to staying (sum of all 5 paths equals 1 again)
            pr[i] = 0

    # return probability for the target state
    return pr[neighbours.index(s2)]


def get_all_states():
    """
    Return a generator to iterate over all possible states.
    Can be used like this:
    for s in get_all_states():
        (do something with s)
    """
    for x in range(MAZE.shape[0]):
        for y in range(MAZE.shape[1]):
            if is_in_maze((x, y)):
                yield (x, y)

# Implement the following functions regard part 1 of the assignment:


def reward(s, s2, a):
    """ Return the reward for taking action a in state s and ending up in s2. """
    reward = 0
    if MAZE[s2] == GOAL:
        reward = 10
    else:
        pass
    return reward


def value_backup(policy, s, V, discount=0.9):
    """
    Compute and return the new value only for state s using the current values V and the current policy.
    The value backup is related to policy evaluation (see slides 37).
    """
    row, col = s
    neighbours = get_neighbours(s)
    v = 0
    for a in ACTIONS:
        for n in neighbours:
            if is_in_maze(n):
                if MAZE[s] == GOAL:
                    v = 0
                    # print(v)
                    return v

                else:
                    row_n, col_n = n
                    v += policy[s][ACTION_IDX[a]] * (state_prob(s, n, a)
                                                     * (reward(s, n, a) + discount * V[n]))
    return v


def policy_evaluation(policy, discount=0.9, epsilon=1e-9):
    """
    Iteratively computes the values of each state following a specific policy.
    The starting values for each state should be 0 when evaluating a new policy!
    Stop the iterative procedure if no state has a bigger change than epsilon.
    Returns the final values for each state.
    """
    V = np.zeros_like(MAZE, dtype=np.float64)
    #V[4][6] = 10
    # print(V)
    converging = True
    while converging:
        V_temp = np.copy(V)
        for s in get_all_states():
            row, col = s
            V[s] = value_backup(policy, s, V_temp, discount=discount)
        for s in get_all_states():
            row_i, col_i = s
            if abs(V[s] - V_temp[s]) > epsilon:
                converging = True
                break
            else:
                converging = False
    return V


def policy_improvement(policy, V, discount=0.9):
    """
    Update the policy by greedily choosing actions based on the current values of states.
    Returns the new policy
    """

    for s in get_all_states():
        row, col = s
        neighbours = get_neighbours(s)
        max = 0
        argmax = 0
        for n in neighbours:
            if is_in_maze(n):
                row_n, col_n = n
                for a in ACTIONS:
                    max_temp = state_prob(
                        s, n, a) * (reward(s, n, a) + discount * V[n])
                    if max_temp > max:
                        max = max_temp
                        argmax = a
        for a in ACTIONS:
            if a == argmax:
                policy[s][ACTION_IDX[a]] = 1.0
            else:
                policy[s][ACTION_IDX[a]] = 0.0

    return policy


def policy_iteration(discount=0.9):
    """
    Us a random starting policy and iterativly improve it until it is converged.
    Use the functions policy_evaluation and policy_iteration when implementing this function.
    If you copy numpy array make sure to copy by value us np.copy()
    Returns the converged policy and the final values of each state.
    """
    # init random policy
    policy = 1 / len(ACTIONS) * \
        np.ones((MAZE.shape[0], MAZE.shape[1], len(ACTIONS)))
    policy_stable = False
    while not policy_stable:
        V = policy_evaluation(policy, discount=discount)
        # print(V)
        policy_temp = np.copy(policy)
        policy = policy_improvement(policy, V, discount=discount)
        if np.array_equal(policy, policy_temp):
            policy_stable = True

    return policy, V

# Implement the following functions regarding part 2 of the assignment:


START_STATE = (1, 1)


def sample_state(s, a):
    """ Starting from state s and using action a randomly sample the next state. """
    neighbours = get_neighbours(s)
    # print(s)
    # print(neighbours)
    p = []
    # a = a[0]
    for n in neighbours:
        p.append(state_prob(s, n, a))
    # print(p)
    # print(a)
    ind_neighbour = np.random.choice(5, 1, p=p)
    # print(ind_neighbour[0])
    neighbour = neighbours[ind_neighbour[0]]
    # print(neighbour)
    return neighbour


def TD_backup(Q, s1, s2, a, discount, alpha):
    """
    Computes the new state-action value based on the s1,a,s2 transition.
    Similar to the value_backup function.
    """
    max = 0
    argmax = 0
    for act in ACTIONS:
        val_temp = Q[s2][ACTION_IDX[act]]
        if val_temp >= max:
            max = val_temp
            # print(act)
            argmax = act
    if MAZE[s1] == GOAL:
        Q[s1][ACTION_IDX[a]] = 0  # Q[s1][ACTION_IDX[a]]
    else:
        Q[s1][ACTION_IDX[a]] = Q[s1][ACTION_IDX[a]] + alpha * \
            (reward(s1, s2, a) + discount * float(Q[s2][ACTION_IDX[argmax]]) -
             float(Q[s1][ACTION_IDX[a]]))

    # no return


def epsilon_greedy(q, epsilon):
    """
    Select and return the action according to the epsilon-greedy policy.
    Greedy exploration in model-free RL doesn't cover the full possible state space.
    """
    m = epsilon / len(ACTIONS)
    max = 0
    argmax = 0
    for a in ACTIONS:
        val_temp = q[ACTION_IDX[a]]
        if val_temp > max:
            max = val_temp
            argmax = ACTION_IDX[a]
    p = [m + (1 - epsilon) if a == argmax else m for a in ACTIONS]
    action = np.random.choice(ACTIONS, 1, p)
    # print(action[0])
    return action[0]


def run_episode(Q, max_steps=100, discount=0.9, alpha=0.4, epsilon=0.05):
    """
    Run a single episode starting from the start_state position until the agent reaches the goal.
    Returns the accumulated reward.
    Make use of the already existing functions like epsilon_greedy, sample_state, reward, and TD_backup.
    def reward(s, s2, a):
        Return the reward for taking action a in state s and ending up in s2.
        # row, col = s2
        # reward = V[row][col]
        # return reward
        return 0
    """
    start_state = START_STATE
    episode_reward = 0
    while max_steps > 0:
        # print(max_steps)
        action = epsilon_greedy(Q[start_state], epsilon)
        next_state = sample_state(start_state, action)
        # print(next_state)
        #val_temp = Q[start_state][ACTION_IDX[action]]
        TD_backup(Q, start_state, next_state, action, discount, alpha)
        #episode_reward += Q[start_state][ACTION_IDX[action]] - val_temp
        start_state = next_state
        if MAZE[start_state] == GOAL:
            episode_reward += 10
            break
        max_steps -= 1
    return episode_reward


def q_learning(episodes):
    """
    Learns the Q values for the given MAZE environment with Q-Learning using a number of episodes and
    returns Q, the total_reward over all episodes, and the reward each episode.
    """
    Q = np.zeros((MAZE.shape[0], MAZE.shape[1], ACTIONS.shape[0]))
    for a in ACTIONS:
        Q[-1][-1][ACTION_IDX[a]] = 0
    total_reward = 0
    rewards_list = []
    for i in range(0, episodes):
        reward = run_episode(Q)
        total_reward += reward
        rewards_list.append(total_reward)
    reward_over_time = total_reward / episodes
    return Q, total_reward, reward_over_time
