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
        self.env = pyRDDLGym.make('TrafficBLX_SimplePhases', 1) #added an environment, using instance 1 for it's larger horizon
        self.stay = {'advance___i0':0}    #this is dumb but it works for now
        self.change = {'advance___i0':1}
        self.RandomAgent = agent1.RandomAgent(
            action_space=self.env.action_space,
            num_actions=self.env.max_allowed_actions)


    def step(self):   # preforms one step of expansion and simulates it
        node = self.root_node
        state = deepcopy(self.root_state)

        while (len(node.child_stay)+len(node.child_change)) != 0:   #choose the best child and advance state accordingly
            if len(node.child_stay)==0:
                node = node.child_change
                state,_,_,_,_ = self.env.step(self.change)
            elif len(node.child_change)==0:
                node = node.child_stay
                state,_,_,_,_ = self.env.step(self.stay)
            elif node.child_stay.value < node.child_change.value:
                node = node.child_change
                state,_,_,_,_ = self.env.step(self.change)
            else:
                node = node.child_stay
                state,_,_,_,_ = self.env.step(self.stay)

        if node.depth>200 :             #the simulation reached maximal depth
            return "sim depth over"
        else:
            if (state["signal___i0"]%2 == 0)and(state["signal-t___i0"] < 4)or((state["signal___i0"] % 2 == 1) and (state["signal-t___i0"] < 60)) :
                node.child_stay = Node(node)     #create the stay_child
                sim_reward = self.simulate(state)
                self.back_propagate(node,sim_reward)
                return self.stay

            elif ((state["signal___i0"] % 2 == 0) and (state["signal-t___i0"] >= 4))or((state["signal___i0"] % 2 == 1) and (state["signal-t___i0"] >= 6)):
                node.child_change = Node(node)   #create the change_child
                sim_reward = self.simulate(state)
                self.back_propagate(node,sim_reward)
                return self.change

            else:                        #the simulation reached a dead end
                return "reached dead end"



    def simulate(self,state):       #rollout the result to get final cumulative reward
        simulated_reward = 0
        for step in range(1,200):#for 200 step, to simulate the original 200 steps intersection
            action = self.RandomAgent.sample_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            simulated_reward = simulated_reward + reward
            state = next_state
            if truncated or terminated:
                break
        return simulated_reward

    def simulate2(self,state):
        tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)
        agent = agent1.RandomAgent(
            action_space=tmp_env.action_space,
            num_actions=tmp_env.max_allowed_actions)
        cmlt_reward = 0
        for step in range(tmp_env.horizon):
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = tmp_env.step(action)
            cmlt_reward = cmlt_reward + reward
            state = next_state
            if truncated or terminated:
                break
        tmp_env.close()
        return cmlt_reward


    def back_propagate(self, node, reward):
        self.env.reset()#!!!!!!!!!!need to reset to root_state somehow!!!!!!!!!!!!!
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
            return self.change
        else:
            return self.stay


    def statistics(self):      #return the overall calculation statistics
        return self.num_rollouts, self.run_time
