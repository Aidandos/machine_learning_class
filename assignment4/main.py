#
# USI - Universita della svizzera italiana
#
# Machine Learning - Fall 2017
#
# Assignment 3: Reinforcement Learning
#
# Use this file to {run, print, plot} your {agent, results} using the policy iteration
# and q-learning functions from the library file.
#
# (!) We will only run the code in this file if necessary. Please provide all relevant
#     results in your report.
#

from library import *
import matplotlib.pyplot as plt


def run_part1():
    #plt.imshow(MAZE, interpolation='nearest', cmap='gray')
    # plt.show()
    policy, V = policy_iteration(discount=0.7)
    print(np.round(V, 4))
    print(policy.argmax(axis=2).choose(ACTIONS))
    plt.imshow(V, interpolation='nearest', cmap='magma')
    plt.show()

    # (...)


def run_part2():
    Q, total_reward, reward_over_time = q_learning(100)
    MAP = np.zeros_like(MAZE, dtype=np.float64)
    for s in get_all_states():
        MAP[s] = Q[s][np.argmax(Q[s])]
    print(Q.argmax(axis=2).choose(ACTIONS))
    print(MAP)
    print(total_reward, reward_over_time)
    plt.imshow(MAP, interpolation='nearest', cmap='magma')
    plt.show()


# run_part1()
run_part2()
