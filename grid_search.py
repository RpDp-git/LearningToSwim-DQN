import DQN
import torch
from torch.utils.tensorboard import SummaryWriter
from itertools import product

scores=[]
render=False
save=False


parameters = dict(lr = [0.00025, 0.0003, 0.00015],
                tau = [5,10,15,20],
                gamma = [0.9,0.93,0.95,0.96],
                epsdecay = [0.99,0.98,0.999])
param_values = [v for v in parameters.values()]

for lr, tau, gamma, epsdecay in product(*param_values):
    agent=DQN.Pw_Agent(gamma=0.93,epsilon=1.0)
    agent.learning_rate=lr
    agent.Tau=tau
    agent.gamma=gamma
    agent.epsdecay=epsdecay
    vic=[]



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
