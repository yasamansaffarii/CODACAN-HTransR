import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .network import *

use_cuda = torch.cuda.is_available()

class DistributionalDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, duel):
        super(DistributionalDQN, self).__init__()
        
        network = CategoricalDuelNetwork if duel else CategoricalNetwork
        self.v_min = -40
        self.v_max = 80
        self.atoms = 51
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms).unsqueeze(0)
        self.delta = (self.v_max - self.v_min) / float(self.atoms - 1)
        if use_cuda:
            self.support = self.support.cuda()
        #self.model = network(input_size, hidden_size, output_size, self.atoms)
        #self.target_model = network(input_size, hidden_size, output_size, self.atoms)
        self.seq_len=5

        
        if self.seq_len == 3:
            input_sizes=[input_size, input_size, input_size]
        elif self.seq_len == 5:
            input_sizes=[input_size, input_size, input_size, input_size, input_size]
        elif self.seq_len == 7:
            input_sizes=[input_size, input_size, input_size, input_size, input_size, input_size, input_size]
        
        self.model = network(input_sizes, hidden_size, output_size, self.atoms)
        self.critic_model = network(input_sizes, hidden_size, output_size, self.atoms)
        self.critic_model.load_state_dict(self.model.state_dict())
        
        # hyper parameters
        self.max_norm = 1
        lr = 0.001
        self.tau = 1e-2
        self.regc = 1e-3

        #self.optimizer = optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)

        if use_cuda:
            self.cuda()

    def update_fixed_critic_network(self):
        #self.target_model.load_state_dict(self.model.state_dict())
        for critic_param, param in zip(self.critic_model.parameters(), self.model.parameters()):
            critic_param.data.copy_(critic_param.data * (1.0 - self.tau) + param.data * self.tau)

    def Variable(self, x):
        x = x.detach()
        if use_cuda:
            x = x.cuda()
        return x
        #return Variable(x, requires_grad=False).cuda() if use_cuda else Variable(x, requires_grad=False)

    def singleBatch(self, raw_batch, params):
        gamma = params.get('gamma', 0.9)
        batch_size = len(raw_batch)
        batch = [np.vstack(b) for b in zip(*raw_batch)]


        # Dynamically create variables s6, s5, ..., s1
        for i in range(self.seq_len, 0, -1):
            globals()[f"s{i}"] = self.Variable(torch.FloatTensor(batch[self.seq_len - i - 1]))

        # Create remaining variables
        s = self.Variable(torch.FloatTensor(batch[self.seq_len - 1]))
        a = self.Variable(torch.LongTensor(batch[self.seq_len]))
        r = self.Variable(torch.FloatTensor(batch[self.seq_len + 1]))
        s_prime = self.Variable(torch.FloatTensor(batch[self.seq_len + 2]))
        done = self.Variable(torch.FloatTensor(np.array(batch[self.seq_len + 3]).astype(np.float32)))






        '''# each example in a batch: [s2, s1, s, a, r, s_prime, term]
        s2 = self.Variable(torch.FloatTensor(batch[0]))
        s1 = self.Variable(torch.FloatTensor(batch[1]))
        s = self.Variable(torch.FloatTensor(batch[2]))
        a = self.Variable(torch.LongTensor(batch[3]))
        r = self.Variable(torch.FloatTensor(batch[4]))
        s_prime = self.Variable(torch.FloatTensor(batch[5]))
        done = self.Variable(torch.FloatTensor(np.array(batch[6]).astype(np.float32)))'''
        #r = r.clamp(-1, 1)

        with torch.no_grad():
            if self.seq_len == 3:
                prob_next = self.critic_model(s1, s, s_prime).detach()
            elif self.seq_len == 5:
                prob_next = self.critic_model(s3, s2, s1, s, s_prime).detach()
            elif self.seq_len == 7:
                prob_next = self.critic_model(s5, s4, s3, s2, s1, s, s_prime).detach()
            q_next = (prob_next * self.support).sum(-1)
            a_next = torch.argmax(q_next, -1)
            #prob_next = self.target_model(s_prime).detach()
            prob_next = prob_next[list(range(batch_size)), a_next, :]
            
            atom_next = r + gamma * (1 - done) * self.support
            atom_next.clamp_(self.v_min, self.v_max)
            b = (atom_next - self.v_min) / self.delta
            l, u = b.floor(), b.ceil()
            d_m_l = (u + (l == u).float() - b) * prob_next
            d_m_u = (b - l) * prob_next
            critic_prob = self.Variable(torch.zeros(prob_next.size()))
            for i in range(critic_prob.size(0)):
                critic_prob[i].index_add_(0, l[i].long(), d_m_l[i])
                critic_prob[i].index_add_(0, u[i].long(), d_m_u[i])
        if self.seq_len == 3:  # For single-element tensors
            log_prob = self.model(s2, s1, s, log_prob=True)
        elif self.seq_len == 5:
            log_prob = self.model(s4, s3, s2, s1, s, log_prob=True)
        elif self.seq_len ==7:
            log_prob = self.model(s6, s5, s4, s3, s2, s1, s, log_prob=True)
        log_prob = log_prob[list(range(batch_size)), a.squeeze(), :]
        loss = -(critic_prob * log_prob).sum(-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(self.model.parameters(), self.max_norm)
        self.optimizer.step()
        self.update_fixed_critic_network()
        return {'cost': {'loss_cost': loss.item(), 'total_cost': loss.item()}, 'error': 0, 'intrinsic_reward': 0}

    def predict(self, *args, a, predict_model):
        if len(args) == 7:
            inputs = self.Variable(torch.from_numpy(args[6]).float())
            #input2 if isinstance(input2, torch.Tensor) else torch.from_numpy(input2).float()
            input6 = self.Variable(torch.from_numpy(args[5]).float())
            input5 = self.Variable(torch.from_numpy(args[4]).float())
            input4 = self.Variable(torch.from_numpy(args[3]).float())
            input3 = self.Variable(torch.from_numpy(args[2]).float())
            input2 = self.Variable(torch.from_numpy(args[1]).float())
            input1 = self.Variable(torch.from_numpy(args[0]).float())
            #print(type(inputs),type(input2),type(input1) )
            prob = self.model(input6, input5, input4, input3, input2, input1, inputs)
        elif len(args) == 5:
            inputs = self.Variable(torch.from_numpy(args[4]).float())
            #input2 if isinstance(input2, torch.Tensor) else torch.from_numpy(input2).float()
            input4 = self.Variable(torch.from_numpy(args[3]).float())
            input3 = self.Variable(torch.from_numpy(args[2]).float())
            input2 = self.Variable(torch.from_numpy(args[1]).float())
            input1 = self.Variable(torch.from_numpy(args[0]).float())
            #print(type(inputs),type(input2),type(input1) )
            prob = self.model(input4, input3, input2, input1, inputs)
        elif len(args) == 3:
            inputs = self.Variable(torch.from_numpy(args[2]).float())
            #input2 if isinstance(input2, torch.Tensor) else torch.from_numpy(input2).float()
            input2 = self.Variable(torch.from_numpy(args[1]).float())
            input1 = self.Variable(torch.from_numpy(args[0]).float())
            #print(type(inputs),type(input2),type(input1) )
            prob = self.model(input2, input1, inputs)

        q = (prob * self.support).sum(-1)
        return q.max(-1)[1].item()

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print("model saved.")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print("model loaded.")
