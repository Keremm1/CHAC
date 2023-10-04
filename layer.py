import numpy as np
import random

from NN.actornet import SthotasticActorModel
from NN.criticnet import CriticModel
from NN.curiousnet import CuriousModel

from experience_buffer import ExperienceBuffer

from tensorflow import keras


#typing
from Enviroment.env import Enviroment
from agent import HACAgent

class Layer:
    def __init__(self, layer_number, agent):

        self.layer_number : int = layer_number
        
        self.agent : HACAgent = agent

        self.actor = SthotasticActorModel()
        self.critic = CriticModel()

        self.curious_model = CuriousModel()
        #η az ise merak fazla ise dışsal(extrinsic) ödül fazladır 
        self.ita = 0.25

        self.temp_goal_replay_storage = []

        self.goal = None

        self.subgoal_penalty = -5

        self.current_state = None

        self.buffer_size = 10000000
        self.batch_size = 1024
        self.replay_buffer = ExperienceBuffer(self.buffer_size, self.batch_size)

        self.num_replay_goals = 3

        
        if self.layer_number == 0:
            self.noise_perc = 0.1
        else:
            self.noise_perc = 0.03

    def add_noise(self, action):
        # Gaussian gürültü ekleyerek yeni bir liste oluştur
        noisy_action = []
    
        for value in action:
            noisy_value = [v + np.random.normal(0, self.noise_perc) for v in value]
            noisy_action.append(noisy_value)
    
        return noisy_action

    
    def choose_action(self, subgoal_testing):
        #eğer subgoal testing true ise subgoal testing için normal davranış sergilenir. Bu sayede testing doğruluğu artar.
        if subgoal_testing:
            return self.actor(self.current_state, self.goal)

        #eğer subgoal testing false ise keşfi arttırmak için normal davranış üzerine gürültü(noise) eklenir veya random bir aksiyon seçilir
        else:
            #keşfi çok fazla arttırmak yerine %80 ihtimalle daha az random aksiyon seçilir
            if np.random.random_sample() > 0.2:
                return self.add_noise(self.actor(self.current_state, self.goal))
            #keşfi çok daha fazla arttırır
            else:
                return [random.randint(1, 100) for _ in range(31)]

    def is_testing_subgoal(self, subgoal_testing):
        # Gelen goalin subgoal_testingi false ise next_subgoal_test false yani mevcut layerın subgoal testine üst layer karar veriyor
        if subgoal_testing:
            return True
        
        else:
            # %20 ihtimalle subgoal testi yapılır.
            # Eğer hiç yapılmasaydı üst π alt π davranışlarına uymayacaktı.
            # Eğer her zaman yapılacak olursa da keşif azalacaktı.
            if np.random.random_sample() < self.agent.subgoal_test_perc:
                return True
            else:
                return False

    
    def calculate_curious_reward(self, action, current_state, next_state, extrinsic_reward):
        predicted_state = self.curious_model(action, current_state)
        instrinsic_reward = keras.losses.MSE(next_state, predicted_state)/2
        norm_instrinsic_reward = (instrinsic_reward - min(self.replay_buffer.curious_buffer) / (max(self.replay_buffer.curious_buffer) - max(self.replay_buffer.curious_buffer)) - 1)
        self.replay_buffer.add_curious(instrinsic_reward)
        reward = self.ita * norm_instrinsic_reward + (1- self.ita) - extrinsic_reward #exponential_average
        return reward

    def perform_action_replay(self, hindsight_action, next_state, goal_status):
        if goal_status[self.layer_number]:
            reward = self.calculate_curious_reward(hindsight_action, self.current_state, next_state, 0)
            finished = True
        else:
            reward = self.calculate_curious_reward(hindsight_action, self.current_state, next_state, -1)
            finished = False

        #[old state, hindsight_action, reward, next_state, goal, terminate boolean, None]
        transition = [self.current_state, hindsight_action, reward, next_state, self.goal, finished, None]

        self.replay_buffer.add(np.copy(transition))

    
    def create_prelim_goal_replay_trans(self, hindsight_action, next_state):
        #[old state, hindsight action, reward = None, next state, goal = None, finished = None, next state projeted to subgoal/end goal space]
        hindsight_goal = next_state

        transition = [self.current_state, hindsight_action, None, next_state, None, None, hindsight_goal]

        self.temp_goal_replay_storage.append(np.copy(transition))

 
    def penalize_subgoal(self, subgoal, next_state):

        transition = [self.current_state, subgoal, self.subgoal_penalty, next_state, self.goal, True, None]

        self.replay_buffer.add(np.copy(transition))

    def get_reward(self,new_goal, hindsight_goal, goal_thresholds):

        for i in range(len(new_goal)):
            if np.absolute(new_goal[i]-hindsight_goal[i]) > goal_thresholds[i]:
                return -1

        return 0

    def finalize_goal_replay(self,goal_thresholds):

        num_trans = len(self.temp_goal_replay_storage)

        num_replay_goals = self.num_replay_goals
        if num_trans < self.num_replay_goals:
            num_replay_goals = num_trans

        indices = np.zeros((num_replay_goals))
        indices[:num_replay_goals-1] = np.random.randint(num_trans,size=num_replay_goals-1)
        indices[num_replay_goals-1] = num_trans - 1
        indices = np.sort(indices)

        for i in range(len(indices)):
            trans_copy = np.copy(self.temp_goal_replay_storage)

            new_goal = trans_copy[int(indices[i])][6]
            for index in range(num_trans):

                trans_copy[index][4] = new_goal

                trans_copy[index][2] = self.get_reward(new_goal, trans_copy[index][6], goal_thresholds)

                if trans_copy[index][2] == 0:
                    trans_copy[index][5] = True
                else:
                    trans_copy[index][5] = False

                self.replay_buffer.add(trans_copy[index])
        self.temp_goal_replay_storage = []

    
    def train(self, env:Enviroment, subgoal_testing = False) -> tuple[list[bool], int]:
        
        self.goal = self.agent.goal_array[self.layer_number]
        self.current_state = self.agent.current_state

        attempts_made = 0

        max_lay_achieved = None

        self.maxed_out = False
        #Eğer layer max_attempt ulaşmışsa veya üst layerlardan birinin hedefini gerçekleştirmişse veya env max actiona ulaşmışsa
        while not (attempts_made > self.agent.time_limit or max_lay_achieved is not None and max_lay_achieved >= self.layer_number or self.agent.steps_taken >= env.max_actions):
            
            action = self.choose_action(subgoal_testing)

            if self.layer_number > 0:
                
                # Gelen goalin subgoal_testingi false ise next_subgoal_test false yani mevcut layerın subgoal testine üst layer karar veriyor
                next_subgoal_testing = self.is_testing_subgoal(subgoal_testing)

                self.agent.goal_array[self.layer_number - 1] = action
                
                goal_status, max_lay_achieved = self.agent.layers[self.layer_number - 1].train(env, next_subgoal_testing) 

            else:
                next_state = env.step(action)

                hindsight_action = action

                self.agent.steps_taken += 1

                self.agent.current_state = next_state

                # low π goal arraydeki ulaştığı max layer goalini max_lay_achieved'a atar. Herhangi bir layerın subgoalini gerçekleştirdiyse goal_status True olacaktır.
                goal_status, max_lay_achieved = self.agent.check_goals(env)

            attempts_made += 1

            #eğer subgoalin alt π için uygun değilse
            if self.layer_number > 0 and self.agent.layers[self.layer_number-1].maxed_out:
                #alt π'a uyacak bir hedef vermesi için bu verdiği subgoali bir daha vermemesi için cezalandırıyoruz.
                if next_subgoal_testing:
                    #low π'ın high π'a uymasından daha çok high π'ın low π'a uyması önemli. Bu sayede low π'ın keşfi korunmuş oluyor.
                    self.penalize_subgoal(action, self.agent.current_state) 
                #ve alt π hareketini hindsight action olarak alıyoruz. Bu sayede alt π'ın bu davranışının hedefe ulaşıp ulaşmadığını bir daha denemeden anlıyoruz.
                hindsight_action = self.agent.current_state
            
            #eğer subgoalin alt π için uygunsa hindsight actionu subgoal olarak alıyoruz.
            elif self.layer_number > 0 and not self.agent.layers[self.layer_number-1].maxed_out:
                hindsight_action = action
            
            #
            self.perform_action_replay(hindsight_action, self.agent.current_state, goal_status)
            #Hindsight action replay
            self.create_prelim_goal_replay_trans(hindsight_action, self.agent.current_state, self.agent.layers)

            #bunu geç yapmamızın sebebi replay bufferda agent.current_state sonraki durumu tutarken self.current_state ise o anki durumu tutuyor.
            self.current_state = self.agent.current_state

        #eğer low π hedefe uymadıysa subgoal missed bilgisini maxed_out true yaparak üst layera gönderiyoruz.
        if not goal_status[self.layer_number]:
            self.maxed_out = True #missed bilgisi burada tutulur

        #filling HER buffer
        self.finalize_goal_replay(goal_thresholds=1)

        return goal_status, max_lay_achieved #max_lay_achieved döndürme sebebeimiz alt π'ın en yüksek hedefe ulaşıyor olabileceğidir
       

    def learn(self, num_updates:int):
        old_states, actions, rewards, new_states, goals, is_terminals = self.replay_buffer.get_batch()
            