from rddlrepository.core.manager import RDDLRepoManager
manager = RDDLRepoManager(rebuild=True)
import random
import numpy as np
import pyRDDLGym

instance=0
in_arr = {0,1,2,3}
values = [1, 0]
probabilities = [0.3, 0.7]


def test(instance):
    env = pyRDDLGym.make('TrafficBLX_SimplePhases', instance=instance)
    state, _ = env.reset()
    cmlt_reward = 0
    for step in range(env.horizon):
        #env.render(to_display=True)
        variable = random.choices(values, probabilities)[0]
        #action = {'advance___i0':variable}
        action = {'advance___i0': 0}
        _, reward, terminated, truncated, _ = env.step(action)
        cmlt_reward = cmlt_reward + reward
        if truncated or terminated:
            break
    env.close()
    return cmlt_reward

def test_random(instance):
    action = {'advance___i0': 1}
    tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', instance=instance)
    state, _ = tmp_env.reset()
    cmlt_reward = 0
    for step in range(tmp_env.horizon):
        values_4 = [0, 1, 2, 3]
        probabilities_4 = [0.25, 0.25, 0.25, 0.25]
        random_instance = random.choices(values_4, probabilities_4)[0]
        tmp_env = pyRDDLGym.make('TrafficBLX_SimplePhases', instance=random_instance)
        _ = tmp_env.reset()
        tmp_env.set_state(state)
        next_state, reward, terminated, truncated, _ = tmp_env.step(action)
        tmp_env.close()
        state = next_state
        cmlt_reward += reward
    return cmlt_reward


print("reward =", test(instance))

# for instance in in_arr:
#     reward = test(instance)
#     print("instance =", instance, "|| cmlt_reward =", reward)


# rewards_arr = []
# for i in range(50):
#     rewards_arr.append(test(instance))
#
# rewards = np.array(rewards_arr)
# print("instance =", instance, "|| reward_mean =", np.mean(rewards), "|| reward_median =", np.median(rewards))
