import random
import time
import math
from copy import deepcopy


explore_c=2
max_reward = 20000

class Node:
    def __init__(self, action, parent):
        self.action = action
        self.parent = parent
        self.N = 0
        self.total_reward = 0
        self.child_stay = {}
        self.child_change = {}
        self.sim_reward = 0  #simulation reward


    def value(self, explore=explore_c):  #calculate the UCB
        return max_reward-(self.total_reward / self.N) + explore * math.sqrt(math.log(self.parent.N) / self.N)


class MCTS:
    def __init__(self, state):
        self.root_state = deepcopy(state)
        self.root = Node(None, None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0

    def mcts_step(self):   # preforms one step of expansion and simulates it
        node = self.root
        state = deepcopy(self.root_state)

        while len(node.child_stay) != 0:   #choose best child and advance state accordingly
            if node.child_stay.value < node.child_change.value:
                node = node.child_change
                state.advance(node.action)
            else:
                node = node.child_stay
                state.advance(node.action)

        if(state.time_counter>200):
            return "sim over"

        if posible :
            parent.child_stay = Node("stay", parent)
            simulate
            back_propagate

        if posible:
            parent.child_stay = Node("change", parent)
            simulate
            back_propagate

    def simulate(self,state):



    def back_propagate(self, node: Node, reward):

        while node is not None:
            node.N += 1
            node.total_reward += reward
            node = node.parent



    def search(self, time_limit: int):
        start_time = time.process_time()

        num_rollouts = 0
        while time.process_time() - start_time < time_limit:
            self.mcts_step()
            num_rollouts += 1

        run_time = time.process_time() - start_time
        self.run_time = run_time
        self.num_rollouts = num_rollouts


    def best_move(self):
        if self.root.child_stay.value < self.root.child_change.value:
            return "change"
        else:
            return "stay"


    def move(self, move):
        if move in self.root.children:
            self.root_state.move(move)
            self.root = self.root.children[move]
            return

        self.root_state.move(move)
        self.root = Node(None, None)

    def statistics(self) -> tuple:
        return self.num_rollouts, self.run_time
