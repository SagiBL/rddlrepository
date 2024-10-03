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
for step in range(env.horizon):
    env.render()
    action = agent.sample_action(state,cmlt_reward)
    next_state, reward, terminated, truncated, _ = env.step(action)
    cmlt_reward = cmlt_reward + reward
    state = next_state
    #print(state["signal___i0"],state["signal-t___i0"])
    #print(state['Nc___l0'])
    if truncated or terminated:
        break

print(f'Episode ended with cumulative reward {cmlt_reward}')
env.close()
