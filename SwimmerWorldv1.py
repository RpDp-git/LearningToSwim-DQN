import numpy as np
import matplotlib.pyplot as plt
import math

class Swimmer:

    def __init__(self):
        self.conversion=1
        while True:
            self.x = np.random.uniform(7,36)*self.conversion
            self.y = np.random.uniform(7,36)*self.conversion
            self.state = self.find_state(np.array([self.x,self.y]))
            if self.state!=40:
                break
        self._laserx,self._lasery=0,0
        self.reward = 0
        self.reward_history = 0
        self.loc_history=[]
        self.loc_history.append([self.x,self.y,self._laserx,self._lasery])
        self.vavg=2*self.conversion
        self.targetx=33*self.conversion
        self.targety=33*self.conversion
        self.done=False
        self.dS = self.get_dS([self.x,self.y])
        self.initdS=self.dS
        self.reason='Max Steps'

    def observation(self):
        return [self.x,self.y,self.dS]

    def get_dS(self,a):
        dx=a[0]-self.targetx
        dy =a[1]-self.targety
        return math.sqrt(dx**2 + dy**2)



    def find_state(self,position,width=6):
        a = math.floor((position[0])/width)
        b = math.floor((position[1])/width)
        return int(7)*b + a

    def reward_map(self,cur_state):
        k=-100
        rewards = np.zeros((49,1)) - 1 #Other States
        rewards[:6] = k # Boundary States
        rewards[42:48] = k
        for i in range(7):
            rewards[7*i]= k
            rewards[(7*i)+6] = k
        rewards[40]=100 # Goal state
        return rewards[cur_state][0]


    def action(self,choice):
        dt = 0.05
        conversion=1e-6                                    #iteration time of the camera [s]                        #maximum propulsion velocity [m/s]
        pxtomum=0.0533e-6
        offset=16*0.0533e-6*1e6
        kB = 1.38064852e-23
        diameter=2.19e-6
        radius=diameter/2
        T = 273.15 + 20
        eta=2.414e-5*10**(247.8/(T-140)) #eta=0.001
        D_0 = kB*T/(6*np.pi*eta*radius)
        prefactor=np.sqrt(2*D_0*dt)*1e6
        stepsexp = 2
        particles = np.zeros((stepsexp,6)) #particle       0 - x; 1 - y; 2 - time; 3 - distance to laser 4 - velocity, 5 - angle
        noisesteps = np.zeros((stepsexp,2))
        particles[0,:2]=np.array([self.x,self.y])
        laser = np.zeros((stepsexp+1,2))  #laserposition  0 - x; 1 - y
        deltas = np.zeros(2)
        angles = np.linspace(0,3,4) * np.pi/2 #Action angles
        diffusion=True
        self.reward=0


        #Create random steps
        for i in range(2): noisesteps[:,i] = prefactor*np.random.normal(0,1,stepsexp)

        #Simulate till state_change
        for i in range(1,stepsexp):
            particles[i,2] = particles[i-1,2] + dt                                           #Time point
            particles[i,:2] = particles[i-1,:2]#Position equals position before
            prev_state=self.state


            if diffusion == True:
                particles[i,:2] += noisesteps[i,:]

            # Place laser
            laser[i,0]=particles[i-1,0]+(offset*np.cos(angles[choice]))
            laser[i,1]=particles[i-1,1]+(offset*np.sin(angles[choice]))
            self._laserx=laser[i,0]
            self._lasery=laser[i,1]

            for j in range(2): deltas[j] = particles[i,j]-laser[i,j]                         #calc x/y distance between laser and particle
            particles[i,3] = np.sqrt(pow(deltas[0],2)+pow(deltas[1],2))                      #distance between laser and the particle
            particles[i,4] = self.vavg                                        #look up the velocity according to the distance
            for j in range(2): particles[i,j] += particles[i,4]/particles[i,3]*deltas[j]*dt


            self.state = self.find_state(position=particles[i,:2])
            _dS = self.get_dS(particles[i,:2])

            self.reward = -_dS/self.initdS * 10

            self.x = particles[i,0]
            self.y = particles[i,1]
            self.dS = self.get_dS([self.x,self.y])
            self.loc_history.append([self.x,self.y,self._laserx,self._lasery])

            if self.reward_map(self.state)== 100 or self.reward_map(self.state)== -100 :
                if self.reward_map(self.state)== 100 :
                    self.reason = 'GOAL'
                else :
                    self.reason = 'DEAD'
                self.reward += self.reward_map(self.state)*100
                self.done=True
