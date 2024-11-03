from rddlrepository.core.manager import RDDLRepoManager
manager = RDDLRepoManager(rebuild=True)
import agent1
import pyRDDLGym.core.policy

env = pyRDDLGym.make('TrafficBLX_SimplePhases', 0)

agent = agent1.mcts_Agent(
    action_space=env.action_space,
    num_actions=env.max_allowed_actions)

cmlt_reward = 0
state, _ = env.reset()
tmp_state = state
for step in range(1,50):
    env.render(to_display=True)
    action = agent.sample_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    cmlt_reward = cmlt_reward + reward
    state = next_state
    if step == 30:
        tmp_state = state
    if truncated or terminated:
        break
state, _ = env.reset()
state = tmp_state
env.set_state(state)
for step in range(1,50):
    env.render(to_display=True)
    action = agent.sample_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    cmlt_reward = cmlt_reward + reward
    state = next_state
    if truncated or terminated:
        break
print(f'Episode ended with cumulative reward {cmlt_reward}')
env.close()
