import DQN
import torch
from torch.utils.tensorboard import SummaryWriter

vic=[]
scores=[]
render=False
save=False
agent=DQN.Pw_Agent(gamma=0.93,epsilon=1.0)
tb = SummaryWriter(comment=agent.get_hyperparams())
print(agent.get_hyperparams())
print()
for i in range(400):
    if i%50==0 :
        render=False
    else:
        render=False
    if i%50==0:
        torch.save(agent.policy_NN.state_dict(), 'episode {}'.format(i))


    reason, score,loss = agent.DQNepisode(N_steps=10000,vid=render)
    scores.append(score)
    print("Episode ",i,"Score ",score,'Status: ',reason)
    vic.append(reason)
    if len(vic)>50:
        a=vic[-50:]
        my_dict = {i:a.count(i) for i in a}
        suc=int(my_dict['GOAL'])/50 * 100
        print('success rate = ', suc)
        tb.add_scalar('Success Rate', suc, i)
        tb.flush()

tb.close()
