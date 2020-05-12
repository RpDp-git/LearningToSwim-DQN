
import matplotlib.pyplot as plt
from random import randint,random,sample
import numpy as np
from math import atan,sin,cos,sqrt,ceil,floor,log
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import SwimmerWorldv1
import visualize as env

class DQN(nn.Module):

    def __init__(self,D_in,H,D_out):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(D_in,H)
        self.lin2 = nn.Linear(H,D_out)

    def forward(self, x):
        out = self.lin2(torch.tanh(self.lin1(x)))
        return(out)


class Pw_Agent:

    def __init__(self,**kwargs):

        self.agent = SwimmerWorldv1.Swimmer()

        self.gamma = kwargs.get('gamma',1.0)
        self.epsilon = kwargs.get('epsilon',0.0)
        self.tau=11

        self.N_batch = 50

        self.initLearningParams()


    def initLearningParams(self):

        self.dtype = torch.float64
        self.device = torch.device("cuda")

        torch.set_default_dtype(self.dtype)
        torch.set_default_tensor_type(torch.DoubleTensor)

        D_in, H, D_out = 3, 100, 4
        self.policy_NN = DQN(D_in,H,D_out)
        self.target_NN = DQN(D_in,H,D_out)
        self.target_NN.load_state_dict(self.policy_NN.state_dict())
        self.target_NN.eval()
        self.optimizer = optim.RMSprop(self.policy_NN.parameters(),lr=0.001)
        self.samples_Q = []


    def updateTargetNetwork(self):
        self.target_NN.load_state_dict(self.policy_NN.state_dict())

    def resetStateValues(self):
        self.agent.__init__()

    def forwardPassQ(self,state_vec):
        Q_s_a = self.policy_NN(state_vec)
        return(Q_s_a)

    def forwardPassQFrozen(self,state_vec):
        Q_s_a = self.target_NN(state_vec)
        return(Q_s_a)


    def singleStateForwardPassQ(self,state_vec):
        qsa = torch.squeeze(self.forwardPassQ(torch.unsqueeze(torch.Tensor(state_vec),dim=0)))
        return(qsa)


    def greedyAction(self,state_vec):
        qsa = self.singleStateForwardPassQ(state_vec)
        return(torch.argmax(qsa))


    def epsGreedyAction(self,state_vec):
        if random()>self.epsilon:
            return(self.greedyAction(state_vec))
        else:
            return(self.getRandomAction())

    def getRandomAction(self):
        return(randint(0,3))

    def DQNepisode(self,save_chkpt=False,N_steps=10**4,vid=False):

        
        R_tot = 0
        self.agent.__init__()

        s = self.agent.observation()
        a = self.epsGreedyAction(s)
        
        self.agent.action(a)
        r = self.agent.reward

        for i in range(N_steps):

            if i%self.tau==0 and i>self.N_batch:
                self.updateTargetNetwork()

            
            R_tot += r
            s_next = self.agent.observation()
            a_next = self.epsGreedyAction(s_next)

            experience = (s,a,r,s_next)
            self.samples_Q.append(experience)

            if len(self.samples_Q)>=2*self.N_batch:

                #Get random batch
                batch_Q_samples = sample(self.samples_Q,self.N_batch)
                states = torch.Tensor(np.array([samp[0] for samp in batch_Q_samples]))
                actions = [samp[1] for samp in batch_Q_samples]
                rewards = torch.Tensor([samp[2] for samp in batch_Q_samples])
                states_next = torch.Tensor([samp[3] for samp in batch_Q_samples])

                #Get current Q value and target value
                Q_cur = self.forwardPassQ(states)[list(range(len(actions))),actions]
                Q_next = torch.max(self.forwardPassQFrozen(states_next),dim=1)[0]

                TD0_error = F.smooth_l1_loss(Q_cur,(rewards + self.gamma*Q_next).detach())

                self.optimizer.zero_grad()
                TD0_error.backward()
                for param in self.policy_NN.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

            s = s_next
            a = a_next
            self.agent.action(a)
            

            if self.agent.done:

                break
        #print('puck-target dist: {:.2f}, R_tot/N_steps: {:.2f}'.format(self.agent.dS,R_tot/N_steps))
        if vid :
            env.render(self.agent.loc_history)
        return(R_tot,self.agent.reason)
