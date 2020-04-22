# LearningToSwim-DQN
Training an RL agent to swim at low Reynolds Number with DeepQLearning and experience replay. Microswimmers are tiny(um-scale) particles that are a simple model to study motion at the microscopic level. In low Reynolds numbers environments the drag forces and brownian motion dominate. Microorganisms like bacteria and fungi has evolved to overcome such hurdles and in many cases use them to their advantage to propel and move around.The microwsimmer (here) is propelled by a laser focused on its edges giving it an average velocity in a particular directions relative to the laser point. 

The environment is written in Pygame and many physical parameters like the diffusion coefficient can be found in the class Swimmer. The observation space is continous and has x, y and the distance from the target as its coordinates. The action space has four elements Up,Left,Right,and Down. The agent recieves a reward when it gets closer to the target and a bigger reward when it actually reaches the target. Each episode starts with the swimmer at a random position and ends when it reaches the target/goes out of bounds. Respectable performance is achieved in 400 episodes, where it consecutively reaches the target more than 20 times. 

Written in pyTorch.

![Alt Text](training.gif)

