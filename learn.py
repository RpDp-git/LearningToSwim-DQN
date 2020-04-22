import DQN
import torch
scores=[]
render=False
save=False
agent=DQN.Pw_Agent(gamma=0.995,epsilon=1.0)
for i in range(600):
    if i%5==0 :
        render=True
    else:
        render=False
    if i%50==0:
        torch.save(agent.policy_NN.state_dict(), 'episode {}'.format(i))


    score=agent.DQNepisode(N_steps=10000,vid=render)
    scores.append(score)
    print("Episode ",i,"Score ",score)
