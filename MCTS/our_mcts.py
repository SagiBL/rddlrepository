import random
import time
import math
from copy import deepcopy
import pyRDDLGym
import agent1

explore_c=2
max_reward = 17644.851216448555
sim_reward = 0

class Node:    #define the format of the nodes
    def __init__(self, parent):
        self.parent = parent
        self.N = 0
        self.total_reward = 0
        self.child_stay = None
        self.child_change = None
        self.depth = parent.depth + 1


    def value(self, explore=explore_c):  #calculate the UCB
        return -max_reward+(self.total_reward / self.N) + explore * math.sqrt(math.log(self.parent.N) / self.N)


class MCTS:
    def __init__(self, state):    #initialize the tree
        self.root_state = deepcopy(state)
        self.root_node = Node(None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0
        self.env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0) #added an environment
        action0 = {} #this is dumb but it works for now
        action0['advance___i0'] = 0
        action1 = {}
        action1['advance___i0'] = 1
        self.stay = action0
        self.change = action1
        self.RandomAgent = agent1.RandomAgent(
            action_space=self.env.action_space,
            num_actions=self.env.max_allowed_actions)


    def step(self):   # preforms one step of expansion and simulates it
        node = self.root_node
        state = deepcopy(self.root_state)

        while (len(node.child_stay)+len(node.child_change)) != 0:   #choose best child and advance state accordingly
            if len(node.child_stay)==0:
                node = node.child_change
                state = self.advance(state, "change")
            elif len(node.child_change)==0:
                node = node.child_stay
                state = self.advance(state, "stay")
            elif node.child_stay.value < node.child_change.value:
                node = node.child_change
                state = self.advance(state,"change")
            else:
                node = node.child_stay
                state = self.advance(state,"stay")

        if node.depth>200 :             #the simulation reached maximal depth
            return "sim depth over"
        else:
            if (state["signal___i0"]%2 == 0)and(state["signal-t___i0"] < 4)or((state["signal___i0"] % 2 == 1) and (state["signal-t___i0"] < 60)) :
                node.child_stay = Node(node)     #create the stay_child
                sim_reward = self.simulate(node,state)
                self.back_propagate(node,sim_reward)
                return "stay"

            elif ((state["signal___i0"] % 2 == 0) and (state["signal-t___i0"] >= 4))or((state["signal___i0"] % 2 == 1) and (state["signal-t___i0"] >= 6)):
                node.child_change = Node(node)   #create the change_child
                sim_reward = self.simulate(node,state)
                self.back_propagate(node,sim_reward)
                return "change"

            else:                        #the simulation reached a dead end
                return "reached dead end"



    def advance(self,state,action):     #advance the state one step according to the action
        if action == "stay":
            action = self.stay
        else:
            action = self.change
        next_state = self.env.step(action)
        state = next_state
        return state


    def simulate(self,node,state):       #rollout the result to get final cumulative reward
        sim_reward = 0
        for step in range(self.env.horizon):
            action = self.RandomAgent.sample_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            sim_reward = sim_reward + reward
            state = next_state
            if truncated or terminated:
                break
        return sim_reward


    def back_propagate(self,node, reward):
        while node is not None:
            node.N += 1
            node.total_reward += reward
            node = node.parent



    def search(self, time_limit: int):
        start_time = time.process_time()

        while time.process_time() - start_time < time_limit:
            self.step()
            self.num_rollouts += 1

        self.run_time = time.process_time() - start_time


    def best_action(self):    #returns the best move for the next iteration
        if self.root_node.child_stay.value < self.root_node.child_change.value:
            return "change"
        else:
            return "stay"


    def statistics(self):      #return the overall calculation statistics
        return self.num_rollouts, self.run_time
