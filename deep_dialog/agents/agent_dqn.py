'''
Created on Jun 18, 2016

An DQN Agent

- An DQN
- Keep an experience_replay pool: training_data <State_t, Action, Reward, State_t+1>
- Keep a copy DQN

Command: python .\run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path .\deep_dialog\data\movie_kb.1k.json --dqn_hidden_size 80 --experience_replay_pool_size 1000 --replacement_steps 50 --per_train_epochs 100 --episodes 200 --err_method 2


@author: xiul
'''


import random, copy, json
import pickle as pickle
import numpy as np
import torch
from collections import deque

from deep_dialog import dialog_config

from .agent import Agent
from deep_dialog.qlearning import DQN, DistributionalDQN
from .prioritized_memory import *


class AgentDQN(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.seq_len = 5
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(list(act_set.keys()))
        self.slot_cardinality = len(list(slot_set.keys()))
        
        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)
        
        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.per = params['per']
        if params['per']:
            self.experience_replay_pool = Memory(params['experience_replay_pool_size'])
        else:
            self.experience_replay_pool = deque(
                maxlen=params['experience_replay_pool_size'])
        
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)
        
        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        
        if params['distributional']:
            #self.dqn = DistributionalDQN(self.state_dimension, self.hidden_size, self.num_actions, params['dueling_dqn'])
            self.dqn = DistributionalDQN(self.state_dimension, self.hidden_size, self.num_actions, params['dueling_dqn'])

        else:
            self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions, params['dueling_dqn'],
                params['double_dqn'], params['icm'], params['noisy'])
        
        self.cur_bellman_err = 0
                
        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            model = torch.load(params['trained_model_path'])
            #self.dqn.model.load_state_dict(model['model'].state_dict())
            self.dqn.load_state_dict(model['state_dict'])
            self.predict_mode = True
            self.warm_start = 2

    def reset_replay(self):    
        if self.per:
            self.experience_replay_pool = Memory(self.experience_replay_pool_size)
        else:
            self.experience_replay_pool = deque(maxlen=self.experience_replay_pool_size)
            
    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """
        
        self.current_slot_id = 0
        self.phase = 0
        
        self.current_request_slot_id = 0
        self.current_inform_slot_id = 0

        self.returns = [[], []]
        
        #self.request_set = dialog_config.movie_request_slots #['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
        self.seq=[]
    def initialize_config(self, req_set, inf_set):
        """ Initialize request_set and inform_set """
        
        self.request_set = req_set
        self.inform_set = inf_set
        self.current_request_slot_id = 0
        self.current_inform_slot_id = 0
        
    def state_to_action(self, state):
        """ DQN: Input state, output action """
        
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
    
    def state_to_action1(self, state):
        """ DQN: Input state, output action """
        
        self.representation = self.prepare_state_representation(state)
        #print(type(self.representation))
        self.seq.append(self.representation)
        # Ensure self.seq always contains 3 elements
        if len(self.seq) < self.seq_len:
            # Add zero vectors to the start of self.seq until its length is 3
            while len(self.seq) < self.seq_len:
                zero_vector = np.zeros_like(self.representation)  # Create a zero vector with the same shape as self.representation
                self.seq.insert(0, zero_vector)  # Add the zero vector to the start of self.seq
        elif len(self.seq) > self.seq_len:
            # Keep only the last 3 elements in self.seq
            self.seq = self.seq[-self.seq_len:]
        #print('ffffff',self.seq[0].shape, self.seq[1].shape, self.representation.shape)#(1, 276) (1, 276) (1, 276)
        #self.action = self.run_policy1(self.seq[0], self.seq[1], self.representation)
        # Dynamically pass elements of self.seq to the function
        self.action = self.run_policy1(*self.seq[:self.seq_len-1], representation=self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

    def get_intrinsic_reward(self, s_t, s_t1, a):
        return self.dqn.get_intrinsic_reward(
                self.prepare_state_representation(s_t),
                self.prepare_state_representation(s_t1),
                np.array(self.action))
    
    def prepare_state_representation(self, state):
        """ Create the representation for each state """
        
        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']
        
        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep =  np.zeros((1, self.act_cardinality))
        user_act_rep[0,self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in list(user_action['inform_slots'].keys()):
            user_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in list(user_action['request_slots'].keys()):
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1,self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in list(agent_last['inform_slots'].keys()):
                agent_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in list(agent_last['request_slots'].keys()):
                agent_request_slots_rep[0,self.slot_set[slot]] = 1.0
        
        turn_rep = np.zeros((1,1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep, agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation

    def create_sequence_pool(self, normal_pool, sequence_length, state_shape):
        """
        Create a sequence-based pool from a normal experience replay pool.
        """
        seq_pool = []
        zero_state = np.zeros(state_shape)

        for i in range(len(normal_pool)):
            if i == 0 or normal_pool[i - 1][4]:  # Check for new episode
                prev_states = [zero_state] * (sequence_length - 1)
            else:
                prev_states = prev_states[1:] + [normal_pool[i - 1][0]]

            current_state, action, reward, next_state, done = normal_pool[i]
            sequence = prev_states + [current_state, action, reward, next_state, done]
            seq_pool.append(sequence)

        return seq_pool

    def run_policy(self, representation):
        """ epsilon-greedy policy """
        
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_request_inform_policy() 
                #return self.rule_policy()
            else:
                return self.dqn.predict(representation, {}, predict_model=True)
                #a, q = self.dqn.predict(representation, {}, predict_model=True, get_q=True)
                #self.returns[1].append(q)
                return a

    def run_policy1(self, *args, representation):
        """ epsilon-greedy policy """
        #print(len(args))
        if len(args) == 7-1:
            rep6, rep5, rep4, rep3, rep2, rep1 = args
        elif len(args) == 3-1:
            #print("yyy")
            rep2, rep1 = args
        elif len(args) == 5-1:
            rep4, rep3, rep2, rep1 = args

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_request_inform_policy() 
                #return self.rule_policy()
            else:
                #print("yyyyy",rep2.shape, rep1.shape, representation.shape)#(1, 276) (1, 276) (1, 276)
                if len(args) == 7-1:
                    a = self.dqn.predict(rep6, rep5, rep4, rep3 ,rep2, rep1, representation, a=None, predict_model=True)
                elif len(args) ==5-1:
                    a = self.dqn.predict(rep4, rep3 ,rep2, rep1, representation, a=None, predict_model=True)
                elif len(args) ==3-1:
                    a = self.dqn.predict(rep2, rep1, representation, a=None, predict_model=True)
  
                #a, q = self.dqn.predict(representation, {}, predict_model=True, get_q=True)
                #self.returns[1].append(q)
                return a
    
    def rule_policy(self):
        """ Rule Policy """
        
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {} }
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {} }
                
        return self.action_index(act_slot_response)
    
    def rule_request_inform_policy(self):
        """ Rule Request and Inform Policy """
        
        if self.current_request_slot_id < len(self.request_set):
            slot = self.request_set[self.current_request_slot_id]
            self.current_request_slot_id += 1

            act_slot_response = {} 
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.current_inform_slot_id < len(self.inform_set):
            slot = self.inform_set[self.current_inform_slot_id]
            self.current_inform_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {slot: 'PLACEHOLDER'}
            act_slot_response['request_slots'] = {}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}
        #else:
        #    raise Exception("THIS SHOULD NOT BE POSSIBLE (AGENT CALLED IN UNANTICIPATED WAY)")
        
        return self.action_index(act_slot_response) #{'act_slot_response': act_slot_response, 'act_slot_value_response': None}
        
    
    def action_index(self, act_slot_response):
        """ Return the index of action """
        
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print(act_slot_response)
        raise Exception("action index not found")
        return None
    
    
    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """ Register feedback from the environment, to be stored as future training data """
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)
        if isinstance(self.experience_replay_pool, Memory):
            target = self.dqn.model(self.dqn.Variable(torch.FloatTensor(state_tplus1_rep))).data
            old_val = target[0][action_t].data
            target_q = self.dqn.target_model(self.dqn.Variable(torch.FloatTensor(state_tplus1_rep)))
            target[0][action_t] = reward_t + self.gamma * (1 - episode_over) * target_q.max()
            err = torch.abs(old_val - target[0][action_t]).item()
            training_example = (err, training_example)
        
        if self.predict_mode == False: # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else: # Prediction Mode
            self.experience_replay_pool.append(training_example)
    
    def train(self, batch_size=1, num_batches=100):
        """ Train DQN with experience replay """
        
        for iter_batch in range(num_batches):
            self.cur_bellman_err = 0
            intrinsic_reward = 0
            if isinstance(self.experience_replay_pool, Memory):
                batch, idx, _ = self.experience_replay_pool.sample(batch_size)
            else:
                first_state = self.experience_replay_pool[0][0]
                state_shape = first_state.shape
                seq_pool=self.create_sequence_pool(self.experience_replay_pool, self.seq_len, state_shape)
                batch = [random.choice(seq_pool) for i in range(batch_size)]
                #print(batch[0])
            batch_struct = self.dqn.singleBatch(batch, {'gamma': self.gamma})
            self.cur_bellman_err += batch_struct['cost']['total_cost']
            intrinsic_reward += batch_struct['intrinsic_reward']
            if isinstance(self.experience_replay_pool, Memory):
                err = batch_struct['error']
                for i in range(batch_size):
                    self.experience_replay_pool.update(idx[i], err[i])
            return self.cur_bellman_err, intrinsic_reward
            print(("cur bellman err %.4f, experience replay pool %s" % (float(self.cur_bellman_err)/len(self.experience_replay_pool), len(self.experience_replay_pool))))
            
    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """
        
        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print('saved model in %s' % (path, ))
        except Exception as e:
            print('Error: Writing model fails: %s' % (path, ))
            print(e)         
    
    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""
        
        self.experience_replay_pool = pickle.load(open(path, 'rb'))
    
             
    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """
        
        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']
        
        print("trained DQN Parameters:", json.dumps(trained_file['params'], indent=2))
        return model
