#  Sanity check
#
# Documentation at https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
# Check from https://gym.openai.com/docs/

import gym
#env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0')
env.reset()

def sanity_check():
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action

def sanity_check_with_observations():
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print(("Episode finished after {} timesteps\n".format(t+1)))
                break

def introspect_env():
    print((env.action_space))
    print((env.observation_space))
    """
        Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

    """

    print((env.observation_space.high))
    print((env.observation_space.low))
    space = gym.spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
    x = space.sample()
    assert space.contains(x),  'Space does not contain x: ' + x
    # assert space.n == 7, 'Space does not contain 7 elements' 
    assert space.n == 8


def main():
    descr = """=========== Let's Play! =========="""
    print(descr)
    # sanity_check()
    # sanity_check_with_observations()
    introspect_env()


if __name__ == "__main__":
    main()
