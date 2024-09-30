class Node:
    def __init__(self, action, parent):
        self.action = action
        self.parent = parent
        self.N = 0
        self.reward = 0
        self.children = {}
        self.outcome = GameMeta.PLAYERS['none']
        self.depth = parent.depth + 1

    def add_children(self, children: dict) -> None:
        for child in children:
            self.children[child.move] = child

    def value(self, explore: float = MCTSMeta.EXPLORATION): #calculates the UCT of the node to see if we should explore this one or another
        if self.N == 0:
            # we prioritize nodes that are not explored
            return 0 if explore == 0 else GameMeta.INF
        else:
            return self.Q / self.N + explore * math.sqrt(math.log(self.parent.N) / self.N)

    def select_node(self) -> tuple:
        node = self.root
        state = deepcopy(self.root_state)

        while len(node.children) != 0:
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value()).value()
            # select nodes with the highest UCT value
            max_nodes = [n for n in children if n.value() == max_value]

            # randomly select on to expand upon
            node = random.choice(max_nodes)
            state.move(node.move)

            if node.N == 0:
                return node, state

        if self.expand(node, state): # determines if the state is a terminal state (game over)
            node = random.choice(list(node.children.values()))
            state.move(node.move)

        return node, state

    def expand(self, parent: Node, state: ConnectState) -> bool:
        if state.game_over():
            return False

        children = [Node(move, parent) for move in state.get_legal_moves()]
        parent.add_children(children)

        return True

    def roll_out(self, state: ConnectState) -> int:
        while not state.game_over():
            state.move(random.choice(state.get_legal_moves()))

        return state.get_outcome()  # function in the game class shown at the bottom of this blog

    def back_propagate(self, node: Node, turn: int, outcome: int) -> None:
        # For the current player, not the next player
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            if outcome == GameMeta.OUTCOMES['draw']:  # we count it as a loss for every state
                reward = 0
            else:
                reward = 1 - reward  # alternates between 0 and 1 because each alternate depth represents different player turns

    def search(self, time_limit: int):
        start_time = time.process_time()

        num_rollouts = 0
        while time.process_time() - start_time < time_limit:
            node, state = self.select_node()
            outcome = self.roll_out(state)
            self.back_propagate(node, state.to_play, outcome)
            num_rollouts += 1  # for calculating statistics

        run_time = time.process_time() - start_time
        self.run_time = run_time
        self.num_rollouts = num_rollouts

    def best_move(self):
        if self.root_state.game_over():
            return -1

        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)

        return best_child.move