import gym
import numpy as np
from misc import FrozenLakeEnv, make_grader
import os
import numpy.random as nr
import matplotlib.pyplot as plt
#%matplotlib inline
np.set_printoptions(precision=3)

env = FrozenLakeEnv()
print(env.__doc__)


class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)
# HG use unwrapped to access 
mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.unwrapped.P.items()}, env.unwrapped.nS, env.unwrapped.nA, env.unwrapped.desc)


#print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
#print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
#print(np.arange(16).reshape(4,4))
#print("Action indices [0, 1, 2, 3] correspond to West, South, East and North.")
#print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
#print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
#print("As another example, state 5 corresponds to a hole in the ice, in which all actions lead to the same state with probability 1 and reward 0.")
#for i in range(4):
#    print("P[5][%i] =" % i, mdp.P[5][i])


def value_iteration(mdp, gamma, nIt, grade_print=print):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)
        
    len(value_functions) == nIt+1 and len(policies) == nIt
    """
    grade_print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    grade_print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)] # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1] # V^{(it)}
        
        # Your code should fill in meaningful values for the following two variables
        # pi: greedy policy for Vprev (not V), 
        #     corresponding to the math above: \pi^{(it)} = Greedy[V^{(it)}]
        #     ** it needs to be numpy array of ints **
        # V: bellman backup on Vprev
        #     corresponding to the math above: V^{(it+1)} = T[V^{(it)}]
        #     ** numpy array of floats **
        
        V = np.zeros(mdp.nS)
        pi = np.zeros(mdp.nS)
        # for each state in the set of states
        for state in mdp.P:
            maxv = 0
            # loop through all the actions in the state
            for action in mdp.P[state]:
                v = 0
                for probability, nextstate, reward in mdp.P[state][action]:
                    v += probability * (reward + gamma * Vprev[nextstate])
                # if this the largest value for this state, update 
                if v > maxv:
                    maxv = v
                    # greedy policy
                    pi[state] = action
            # note above, avoid updating value function in place
            V[state] = maxv

        max_diff = np.abs(V - Vprev).max()
        nChgActions="N/A" if oldpi is None else (pi != oldpi).sum()
        grade_print("%4i      | %6.5f      | %4s          | %5.3f"%(it, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis

GAMMA = 0.95 # we'll be using this same value in subsequent problems


# The following is the output of a correct implementation; when
#   this code block is run, your implementation's print output will be
#   compared with expected output.
#   (incorrect line in red background with correct line printed side by side to help you debug)
expected_output_4 = """Iteration | max|V-Vprev| | # chg actions | V[0]
----------+--------------+---------------+---------
   0      | 0.80000      |  N/A          | 0.000
   1      | 0.60800      |    2          | 0.000
   2      | 0.51984      |    2          | 0.000
   3      | 0.39508      |    2          | 0.000
   4      | 0.30026      |    2          | 0.000
   5      | 0.25355      |    1          | 0.254
   6      | 0.10478      |    0          | 0.345
   7      | 0.09657      |    0          | 0.442
   8      | 0.03656      |    0          | 0.478
   9      | 0.02772      |    0          | 0.506
  10      | 0.01111      |    0          | 0.517
  11      | 0.00735      |    0          | 0.524
  12      | 0.00310      |    0          | 0.527
  13      | 0.00190      |    0          | 0.529
  14      | 0.00083      |    0          | 0.530
  15      | 0.00049      |    0          | 0.531
  16      | 0.00022      |    0          | 0.531
  17      | 0.00013      |    0          | 0.531
  18      | 0.00006      |    0          | 0.531
  19      | 0.00003      |    0          | 0.531"""
Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=20, grade_print=make_grader(expected_output_4))

#env = gym.make('CartPole-v0')
#for i_episode in range(20):
#    observation = env.reset()
#    for t in range(100):
#        env.render()
#        print(observation)
#        action = env.action_space.sample()
#        observation, reward, done, info = env.step(action)
#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break

#for (V, pi) in zip(Vs_VI[:10], pis_VI[:10]):
#    plt.figure(figsize=(3,3))
#    plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
#    ax = plt.gca()
#    ax.set_xticks(np.arange(4)-.5)
#    ax.set_yticks(np.arange(4)-.5)
#    ax.set_xticklabels([])
#    ax.set_yticklabels([])
#    Y, X = np.mgrid[0:4, 0:4]
#    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
#    Pi = pi.reshape(4,4)
#    for y in range(4):
#        for x in range(4):
#            a = Pi[y, x]
#            u, v = a2uv[a]
#            plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1) 
#            plt.text(x, y, str(env.unwrapped.desc[y,x].item().decode()),
#                     color='g', size=12,  verticalalignment='center',
#                     horizontalalignment='center', fontweight='bold')
#    plt.grid(color='b', lw=2, ls='-')
#plt.figure()
#plt.plot(Vs_VI)
#plt.title("Values of different states");

def compute_vpi(pi, mdp, gamma):
    # use pi[state] to access the action that's prescribed by this policy
    a = np.identity(mdp.nS) 
    b = np.zeros(mdp.nS) 
    
    for state in range(mdp.nS):
        for probability, nextstate, reward in mdp.P[state][pi[state]]:
            a[state][nextstate] = a[state][nextstate] - gamma * probability
            b[state] += probability * reward
    
    V = np.linalg.solve(a, b)
    return V

expected_val = np.array([  1.381e-18,   1.844e-04,   1.941e-03,   1.272e-03,   2.108e-18,
         0.000e+00,   8.319e-03,   1.727e-16,   3.944e-18,   2.768e-01,
         8.562e-02,  -7.242e-16,   7.857e-18,   3.535e-01,   8.930e-01,
         0.000e+00])

actual_val = compute_vpi(np.arange(16) % mdp.nA, mdp, gamma=GAMMA)
if np.all(np.isclose(actual_val, expected_val, atol=1e-4)):
    print("Test passed")
else:
    print("Expected: ", expected_val)
    print("Actual: ", actual_val)
    
def compute_qpi(vpi, mdp, gamma):
    Qpi = np.zeros([mdp.nS, mdp.nA]) # REPLACE THIS LINE WITH YOUR CODE
    for state in range(mdp.nS):
        for action in range(mdp.nA):
            for probability, nextstate, reward in mdp.P[state][action]:
                Qpi[state][action] += probability * (reward + gamma * vpi[nextstate]) 
    return Qpi

expected_Qpi = np.array([[  0.38 ,   3.135,   1.14 ,   0.095],
       [  0.57 ,   3.99 ,   2.09 ,   0.95 ],
       [  1.52 ,   4.94 ,   3.04 ,   1.9  ],
       [  2.47 ,   5.795,   3.23 ,   2.755],
       [  3.8  ,   6.935,   4.56 ,   0.855],
       [  4.75 ,   4.75 ,   4.75 ,   4.75 ],
       [  4.94 ,   8.74 ,   6.46 ,   2.66 ],
       [  6.65 ,   6.65 ,   6.65 ,   6.65 ],
       [  7.6  ,  10.735,   8.36 ,   4.655],
       [  7.79 ,  11.59 ,   9.31 ,   5.51 ],
       [  8.74 ,  12.54 ,  10.26 ,   6.46 ],
       [ 10.45 ,  10.45 ,  10.45 ,  10.45 ],
       [ 11.4  ,  11.4  ,  11.4  ,  11.4  ],
       [ 11.21 ,  12.35 ,  12.73 ,   9.31 ],
       [ 12.16 ,  13.4  ,  14.48 ,  10.36 ],
       [ 14.25 ,  14.25 ,  14.25 ,  14.25 ]])

Qpi = compute_qpi(np.arange(mdp.nS), mdp, gamma=0.95)
if np.all(np.isclose(expected_Qpi, Qpi, atol=1e-4)):
    print("Test passed")
else:
    print("Expected: ", expected_Qpi)
    print("Actual: ", Qpi)

def policy_iteration(mdp, gamma, nIt, grade_print=print):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    grade_print("Iteration | # chg actions | V[0]")
    grade_print("----------+---------------+---------")
    for it in range(nIt):        
        # YOUR CODE HERE
        # you need to compute qpi which is the state-action values for current pi
        vpi = compute_vpi(pis[-1], mdp, gamma=gamma)
        qpi = compute_qpi(vpi, mdp, gamma=gamma)
        pi = qpi.argmax(axis=1)
        grade_print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis

expected_output = """Iteration | # chg actions | V[0]
----------+---------------+---------
   0      |      1        | -0.00000
   1      |      9        | 0.00000
   2      |      2        | 0.39785
   3      |      1        | 0.45546
   4      |      0        | 0.53118
   5      |      0        | 0.53118
   6      |      0        | 0.53118
   7      |      0        | 0.53118
   8      |      0        | 0.53118
   9      |      0        | 0.53118
  10      |      0        | 0.53118
  11      |      0        | 0.53118
  12      |      0        | 0.53118
  13      |      0        | 0.53118
  14      |      0        | 0.53118
  15      |      0        | 0.53118
  16      |      0        | 0.53118
  17      |      0        | 0.53118
  18      |      0        | 0.53118
  19      |      0        | 0.53118"""

#Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.65, nIt=20, grade_print=make_grader(expected_output))
#plt.plot(Vs_PI);

'''Q-Learning'''
env = gym.make('FrozenLake-v0')
Q = np.zeros((env.observation_space.n, env.action_space.n))
rewards = []
iterations = []

# Parameters
alpha = 0.55
discount = 0.95
episodes = 1000

# Episodes
for episode in range(episodes):
    # Refresh state
    state = env.reset()
    done = False
    t_reward = 0
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    # Run episode
    for i in range(max_steps):
        if done:
            break

        current = state
        action = np.argmax(Q[current, :] + np.random.randn(1, env.action_space.n) * (1 / float(episode + 1)))

        state, reward, done, info = env.step(action)
        t_reward += reward
        Q[current, action] += alpha * (reward + discount * np.max(Q[state, :]) - Q[current, action])

    rewards.append(t_reward)
    iterations.append(i)

# Close environment
env.close()

# Plot results
def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

size = int(episodes / 50)
chunks = list(chunk_list(rewards, size))
averages = [sum(chunk) / len(chunk) for chunk in chunks]

plt.plot(range(0, len(rewards), size), averages)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()

## Push solution
#api_key = os.environ.get('GYM_API_KEY', False)
#if api_key:
#    print ('Push solution? (y/n)')
#    if raw_input().lower() == 'y':
#        gym.upload(folder, api_key=api_key)