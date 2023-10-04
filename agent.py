from layer import Layer

#typing
from Enviroment.env import Enviroment

class HACAgent:
    def __init__(self, k, time_limit, subgoal_test_perc):
        
        self.subgoal_test_perc = subgoal_test_perc

        self.k = k #layer_count

        self.time_limit = time_limit

        self.layers = [Layer(i, self) for i in range(k)]

        self.current_state = None

        self.goal_array = [None for i in range(k)]
        
        self.steps_taken = 0

        self.num_updates = 40
    

    def train(self, env:Enviroment, episodes:int) -> list[bool]:

        for episode in range(episodes):

            #end goal üst π hedefi olarak goal arraye atanıyor
            self.current_state, self.goal_array[self.k-1] = env.reset()

            goal_status, max_lay_achieved = self.layers[self.k-1].train(env, self.current_state)

            self.learn()

        return goal_status[self.k-1]

    def learn(self):
        for i in range(len(self.layers)):   
            self.layers[i].learn(self.num_updates)

    def check_goals(self, env:Enviroment) -> tuple[list[bool], int]:
        goal_status = [False for i in range(self.k)]

        max_lay_achieved = None

        return goal_status, max_lay_achieved
