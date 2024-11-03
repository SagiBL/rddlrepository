from rddlrepository.core.manager import RDDLRepoManager
manager = RDDLRepoManager(rebuild=True)
import agent1

import pyRDDLGym
import pyRDDLGym.core.policy

env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)

agent = agent1.mcts_Agent(
    action_space=env.action_space,
    num_actions=env.max_allowed_actions)

cmlt_reward = 0
state, _ = env.reset()
for step in range(1,200):
    print("step",step)
    env.render(to_display=True)
    action = agent.sample_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    cmlt_reward = cmlt_reward + reward
    state = next_state
    if truncated or terminated:
        break
print(f'Episode ended with cumulative reward {cmlt_reward}')
env.close()

