import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
import sys
import numpy as np
#sys.path.append('/content/drive/MyDrive/e2e_dialog_challenge/DialogDQN-Variants/deep_dialog/qlearning')
#from global_cosine_attention import GlobalCosineAttention


# Attention mechanism you provided

import torch
import torch.nn as nn
import torch.nn.functional as F
def masked_softmax(vector, mask, dim=-1):
    """
    Apply softmax to the given vector while ignoring masked elements.
    
    Args:
        vector (torch.Tensor): Input tensor to apply softmax.
        mask (torch.Tensor): Binary mask with the same shape as vector.
        dim (int): Dimension to apply softmax.

    Returns:
        torch.Tensor: Masked softmax output.
    """
    mask = mask.float()
    vector = vector * mask
    max_vector = torch.max(vector, dim=dim, keepdim=True)[0]
    vector_exp = torch.exp(vector - max_vector) * mask
    vector_sum = torch.sum(vector_exp, dim=dim, keepdim=True)
    return vector_exp / (vector_sum + 1e-6)
class Hier(nn.Module):
    def __init__(self):
        super(Hier, self).__init__()

    def forward(self, batch_outputs):
        """
        Forward pass for the custom attention layer.
        :param batch_outputs: Batch of outputs from the previous layer, which contains [prev2_state_input, prev_state_input, state_input]
        :return: Attended state tensor
        """
        #print(len(batch_outputs))
        prev2_state_input, prev_state_input, state_input = torch.split(batch_outputs, 1, dim=1)
        '''prev2_state_input = batch_outputs[0]  # Shape: [1, 276]
        prev_state_input = batch_outputs[1]   # Shape: [1, 276]
        state_input = batch_outputs[2]        # Shape: [1, 276]'''
        # Remove the singleton dimension
        prev2_state_input = prev2_state_input.squeeze(1)
        prev_state_input = prev_state_input.squeeze(1)
        state_input = state_input.squeeze(1)
        #print(prev2_state_input.shape,prev_state_input.shape,state_input.shape)#torch.Size([1, 276]) torch.Size([1, 276]) torch.Size([1, 276]) or 1=3 if s_prime_as_s'_from_dist_singlebtch torch.Size([16, 3, 276])

        #Recurrent HIER
        ### Phase 1: Calculate attended embedding for prev states (prev2_state_input and prev_state_input)

        # Cosine similarity between prev2_state_input and prev_state_input (calculate attention for prev states)
        cosine_sim_prev = F.cosine_similarity(prev_state_input, prev2_state_input, dim=-1)  # [1]

        # Apply exponential to cosine similarity to get attention weights
        attention_weights_prev = torch.exp(cosine_sim_prev)  # Attention weight for prev states

        # Normalize the attention weights for prev states
        attention_weights_prev /= torch.sum(attention_weights_prev, dim=-1, keepdim=True)

        # Compute attended embedding for prev states (attended_prev)
        attended_prev = attention_weights_prev.unsqueeze(-1) * prev_state_input  # Attended prev state

        # Add prev_state_input to the attended embedding
        final_prev_state = attended_prev + prev_state_input

        # Normalize the final embedding
        final_prev_state = F.normalize(final_prev_state, p=2, dim=-1)

        ### Phase 2: Use attended prev and calculate attention for state_input

        # Cosine similarity between attended_prev (from Phase 1) and state_input (calculate attention for state_input)
        cosine_sim_state = F.cosine_similarity(attended_prev, state_input, dim=-1)  # [1]

        # Apply exponential to cosine similarity to get attention weights for state_input
        attention_weights_state = torch.exp(cosine_sim_state)  # Attention weight for state_input

        # Normalize the attention weights for state_input
        attention_weights_state /= torch.sum(attention_weights_state, dim=-1, keepdim=True)

        # Compute attended embedding for state_input (attended_state_input)
        attended_state_input = attention_weights_state.unsqueeze(-1) * state_input  # Attended state_input

        # Add state_input to the attended embedding
        final_state_input = attended_state_input + state_input

        # Normalize the final embedding
        final_state_input = F.normalize(final_state_input, p=2, dim=-1)


        #Global HIER
        # Calculate cosine similarity between state_input and prev_state_input
        cosine_sim_state_prev_global = F.cosine_similarity(state_input, prev_state_input, dim=-1)
        attention_weights_state_prev_global = torch.exp(cosine_sim_state_prev_global)
        attention_weights_state_prev_global /= torch.sum(attention_weights_state_prev_global, dim=-1, keepdim=True)

        # Calculate cosine similarity between state_input and prev2_state_input
        cosine_sim_state_prev2_global = F.cosine_similarity(state_input, prev2_state_input, dim=-1)
        attention_weights_state_prev2_global = torch.exp(cosine_sim_state_prev2_global)
        attention_weights_state_prev2_global /= torch.sum(attention_weights_state_prev2_global, dim=-1, keepdim=True)

        # Compute the attended embedding for state_input
        attended_global_state_input = (
            attention_weights_state_prev_global.unsqueeze(-1) * prev_state_input + 
            attention_weights_state_prev2_global.unsqueeze(-1) * prev2_state_input
        )

        # Add state_input to the attended embedding
        final_global_state_input = attended_global_state_input + state_input

        # Normalize the final embedding
        final_global_state_input = F.normalize(final_global_state_input, p=2, dim=-1)



        # Max pooling layer (non-learnable) on final_global_state_input and final_state_input
        final_current_attended_state = torch.max(
            torch.stack([final_state_input, final_global_state_input], dim=0), dim=0
        )[0]  # Perform element-wise max pooling

        # Concatenate final_current_attended_state with state_input
        concatenated_output = torch.cat((final_current_attended_state, state_input), dim=-1)  # Shape: [1, 276 + 276]


        return concatenated_output





'''class TransCAT(nn.Module):
    def __init__(self):
        super(TransCAT, self).__init__()
        # Define state element sizes
        self.state_element_sizes = [11, 31, 31, 31, 11, 31, 31, 1, 34, 32, 32]

    def process_single_input(self, inputs):
        """
        Process a single input tensor with shape [1, 276].
        """
        # Step 1: Split the input tensor
        split_inputs = []
        start_idx = 0
        for size in self.state_element_sizes:
            split_inputs.append(inputs[:, start_idx : start_idx + size])
            start_idx += size

        # Step 2: Pad the split inputs
        max_size = max(self.state_element_sizes)
        padded_inputs = []
        for input_tensor in split_inputs:
            padding_size = max_size - input_tensor.size(1)
            padding = (0, padding_size)  # Pad at the end
            padded_tensor = F.pad(input_tensor, padding, value=1e-6)  # Small padding value
            padded_inputs.append(padded_tensor)

        # Step 3: Stack the padded inputs
        stacked_inputs = torch.stack(padded_inputs, dim=1)  # Shape: (1, num_elements, max_size)
        stacked_inputs = stacked_inputs.squeeze(0)  # Shape: (num_elements, max_size)
        # Debug
        #print(f"Stacked inputs shape (squeezed): {stacked_inputs.shape}")  # Expected: [11, 34]

        # Step 4: Calculate cosine similarity
        num_elements = stacked_inputs.size(0)  # Number of elements (11)
        similarities = torch.zeros(num_elements, num_elements, device=inputs.device)
        #print("A",stacked_inputs.shape)
        for i in range(num_elements):
            for j in range(num_elements):
                #print(stacked_inputs[i].shape,stacked_inputs[j].shape )
                # Ensure both tensors are 1D for cosine similarity
                """similarities[i, j] = F.cosine_similarity(
                    stacked_inputs[i].unsqueeze(0),  # Shape: (1, max_size)
                    stacked_inputs[j].unsqueeze(0),  # Shape: (1, max_size)
                    dim=1  # Cosine similarity across features
                )"""


        # Step 5: Normalize to get attention weights
        attention_weights = F.softmax(similarities, dim=-1)

        # Step 6: Apply attention to the original (unpadded) inputs
        attended_elements = []
        for i in range(num_elements):
            # Apply attention to the padded inputs
            attention_weights_i = attention_weights[i, :].unsqueeze(0)  # Shape: (1, num_elements)
            attended_element = torch.matmul(
                attention_weights_i,  # Shape: (1, num_elements)
                stacked_inputs  # Shape: (num_elements, max_size)
            )  # Shape: (1, max_size)
            attended_elements.append(attended_element[:, : self.state_element_sizes[i]].squeeze(0))

        # Step 7: Concatenate the attended elements and flatten
        flattened_output = torch.cat(attended_elements, dim=0).unsqueeze(0)  # Shape: (1, 276)

        # Debug
        #print(f"Final flattened output shape: {flattened_output.shape}")  # Should match [1, 276]
        return flattened_output

    def forward(self, inputs):
        """
        Forward pass for the TransCAT class.
        Handles inputs of shape [1, 276] or [16, 3, 276].
        """
        #print("inputs.shape",inputs.shape)#torch.Size([1, 3, 276])
        if inputs.dim() == 3 and inputs.size(2) == 276:
            # If inputs are [batch_size=16, seq_length=3, 276], process each sequence individually
            batch_size, seq_length, _ = inputs.size()
            outputs = []
            
            for batch_idx in range(batch_size):
                batch_outputs = []
                for seq_idx in range(seq_length):
                    #print('inputs[batch_idx, seq_idx].unsqueeze(0)',inputs[batch_idx, seq_idx].unsqueeze(0).shape)#torch.Size([1, 276])
                    single_output = self.process_single_input(inputs[batch_idx, seq_idx].unsqueeze(0))  # Process [1, 276]
                    #print('single_output',single_output.shape)#alays torch.Size([1, 276])
                    batch_outputs.append(single_output)
                
                # Combine outputs for this batch
                outputs.append(torch.cat(batch_outputs, dim=0))  # Shape: [seq_length, 276]

            #print('torch.stack(outputs, dim=0)',torch.stack(outputs, dim=0).shape)#torch.Size([3, 16, 276]) or torch.Size([1, 3, 276])
            # Combine all batches
            return torch.stack(outputs, dim=0)'''
        
        """elif inputs.dim() == 2 and inputs.size(1) == 276:
            # If input is [1, 276], process it directly
            return [self.process_single_input(inputs)]
        
        else:
            raise ValueError("Input shape not supported. Expected [1, 276] or [16, 3, 276].")"""


    '''def forward(self, inputs):
        """
        Forward pass for the TransCAT class.
        Handles inputs of shape [1, 276] and [3, 1, 276].
        """
        #print(inputs.shape)
        if inputs.dim() == 3 and inputs.size(0) == 3 and inputs.size(1) == 1:
            # If inputs are [3, 1, 276], process each individually
            outputs = []
            for i in range(3):
                single_output = self.process_single_input(inputs[i])  # Process each [1, 276]
                #print(single_output.shape)
                """torch.Size([1, 276])
                torch.Size([1, 276])
                torch.Size([1, 276])
                outputs.append(single_output)"""

            # Return a list of processed tensors
            return outputs
        elif inputs.dim() == 2 and inputs.size(1) == 276:
            # If input is [1, 276], process it directly
            return [self.process_single_input(inputs)]
        else:
            print("S",inputs.shape)
            raise ValueError("Input shape not supported. Expected [1, 276] or [3, 1, 276].")'''


def init_weight(m, gain=1):
    for name, param in m.named_parameters():
        if name.find('weight') != -1:
            torch.nn.init.xavier_uniform_(param, gain)
        elif name.find('bias') != -1:
            param.data.fill_(0)

'''class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            #nn.init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            #nn.init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            nn.init.constant(self.sigma_weight, self.sigma_init)
            nn.init.constant(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.cuda(), self.bias + self.sigma_bias * self.epsilon_bias.cuda())

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)'''



class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True, noise_prob=0.1):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_prob = noise_prob  # Set the noise probability to 0.1
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            nn.init.constant(self.sigma_weight, self.sigma_init)
            nn.init.constant(self.sigma_bias, self.sigma_init)

    def add_gaussian_noise(self, input_state):
        """
        Adds Gaussian noise to the input state based on mean and standard deviation.

        Parameters:
            input_state (torch.Tensor): The original input state.

        Returns:
            torch.Tensor: The noisy input state.
        """
        mean = input_state.mean()
        std_dev = input_state.std()
        noise = torch.normal(mean=mean, std=std_dev, size=input_state.size()).to(input_state.device)
        noisy_state = input_state + noise
        return noisy_state

    def add_binary_noise(self, input_vector):
        """
        Adds binary noise to the input_vector by randomly setting elements to 0 or 1.

        Parameters:
            input_vector (torch.Tensor): The original input vector.

        Returns:
            torch.Tensor: The noisy input vector.
        """
        noisy_vector = input_vector.clone()

        # Generate a mask for binary noise with the given probability
        binary_noise_mask = torch.rand_like(input_vector) < self.noise_prob

        # Randomly set elements to 0 or 1 based on the binary_noise_mask
        noisy_vector[binary_noise_mask] = torch.randint(0, 2, size=noisy_vector[binary_noise_mask].size()).float()
        
        return noisy_vector

    def forward(self, input):
        noisy_input = self.add_gaussian_noise(input)  # Adding Gaussian noise
        noisy_input = self.add_binary_noise(noisy_input)  # Adding Binary noise
        
        # Perform standard linear transformation
        return super(NoisyLinear, self).forward(noisy_input)


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noisy=False):
        super(Network, self).__init__()
        output_class = NoisyLinear if noisy else nn.Linear
        self.qf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', output_class(hidden_size, output_size))]))
        self.noisy = noisy

    def forward(self, inputs, testing=False):
        return self.qf(inputs)

    def sample_noise(self):
        if self.noisy:
            self.qf.w2.sample_noise()
    
    def remove_noise(self):
        if self.noisy:
            self.qf.w2.remove_noise()
        

class DuelNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noisy=False):
        super(DuelNetwork, self).__init__()
        output_class = NoisyLinear if noisy else nn.Linear
        self.adv = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', output_class(hidden_size, output_size))]))
        self.vf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', output_class(hidden_size, 1))]))
        self.noisy = noisy

    def forward(self, inputs, testing=False):
        v = self.vf(inputs)
        adv = self.adv(inputs)
        return v.expand(adv.size()) + adv - adv.mean(-1).unsqueeze(1).expand(adv.size())

    def sample_noise(self):
        if self.noisy:
            self.adv.w2.sample_noise()
            self.vf.w2.sample_noise()
    
    def remove_noise(self):
        if self.noisy:
            self.adv.w2.remove_noise()
            self.vf.w2.remove_noise()



class CategoricalDuelNetwork(nn.Module):
    def __init__(self, input_size ,hidden_size, output_size, atoms=51):
        super(CategoricalDuelNetwork, self).__init__()
        #restaurant:276 80 54:[11,31,31,31,11,31,31,1,34,32,32]=276
        #taxi:213,80,41:[11,22,22,22,11,22,22,1,34,23,23]
        #movie:272 80 43:[11,29,29,29,11,29,29,1,44,30,30]
        #print(input_size,hidden_size,output_size)
        self.atoms = atoms
        #print("input_size_from_catduelnetwork_network",input_size)#276
        self.output_size = output_size

        #modify per domin
        self.state_representation_size = 276
        self.max_state_length = 34
        self.padding_value = 1e-6
        self.dropout_rate = 0.2
        self.sequence_length=3
        self.state_element_sizes = [11,31,31,31,11,31,31,1,34,32,32]


        # Global Cosine Attention layer
        self.transcat = TransCAT()
        # Initialize the attention layer
        self.hier = Hier()

        self.adv = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(552, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, output_size * atoms))]))
        self.vf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(552, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, atoms))]))

    def forward(self,inputs, testing=False, log_prob=False):

        if inputs.shape == torch.Size([3, 1, 276]):  # Use torch.Size for shape comparison
            # Permute dimensions to achieve shape [1, 3, 276]
            inputs = inputs.permute(1, 0, 2)

        #print("inputs_from_catduelnetwork_network",inputs.shape)#torch.Size([1, 3, 276]) after 17th torch.Size([16, 3, 276]): input from s_prime
        batch_outputs = self.transcat(inputs)
        ####print('batch_outputs', batch_outputs.shape)#torch.Size([1, 3, 276])
        '''for i, output in enumerate(batch_outputs):
            print(f"Batch Output {i+1} Shape_from_catduelnetwork_network: {output.shape}")'''  # Expected: [1, 276]
          #Batch Output Shape: 3th of torch.Size([1, 276]) if inputs shape:torch.Size([1, 3, 276])
          #Batch Output Shape: 16th of torch.Size([3, 276]) if inputs shape:torch.Size([16, 3, 276])


        attended_state = self.hier(batch_outputs)
        #print('attended_state',attended_state.shape)
        v = self.vf(attended_state).view(-1, 1, self.atoms)
        adv = self.adv(attended_state).view(-1, self.output_size, self.atoms)
        #q = adv
        q = v + adv - adv.mean(1, keepdim=True)
        #log prob==>for online net 
        if log_prob:
            #print('log_prob', F.log_softmax(q, -1).shape)
            return F.log_softmax(q, -1)
        else:
            #print('log_prob2', F.softmax(q, -1).shape)#if s_prime_as_s'_from_dist_singlebtch torch.Size([16, 3, 276]) ==torch.Size([1, 54, 51]) after 17TH torch.Size([3, 54, 51])
            return F.softmax(q, -1)

class CategoricalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, atoms=51):
        super(CategoricalNetwork, self).__init__()
        self.atoms = atoms
        self.output_size = output_size
        self.qf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, output_size * atoms))]))

    def forward(self, inputs, testing=False, log_prob=False):
        q = self.qf(inputs).view(-1, self.output_size, self.atoms)
        if log_prob:
            return F.log_softmax(q, -1)
        else:
            return F.softmax(q, -1)

class CategoricalDuelNetwork1(nn.Module):
    def __init__(self, input_size ,hidden_size, output_size, atoms=51):
        super(CategoricalDuelNetwork1, self).__init__()
        self.atoms = atoms
        self.output_size = output_size
        self.dropout_rate = 0.2
        

        self.adv = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, output_size * atoms))]))
        self.vf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, atoms))]))

    def forward(self,inputs, testing=False, log_prob=False):
        #print("B",inputs.shape)
        #print("B",inputs.shape)

        v = self.vf(inputs).view(-1, 1, self.atoms)
        adv = self.adv(inputs).view(-1, self.output_size, self.atoms)
        #q = adv
        q = v + adv - adv.mean(1, keepdim=True)
        if log_prob:
            return F.log_softmax(q, -1)
        else:
            return F.softmax(q, -1)

class CategoricalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, atoms=51):
        super(CategoricalNetwork, self).__init__()
        self.atoms = atoms
        self.output_size = output_size
        self.qf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, output_size * atoms))]))

    def forward(self, inputs, testing=False, log_prob=False):
        q = self.qf(inputs).view(-1, self.output_size, self.atoms)
        if log_prob:
            return F.log_softmax(q, -1)
        else:
            return F.softmax(q, -1)

class CategoricalCritic(nn.Module):
    def __init__(self, input_size, hidden_size, atoms=51, v_min=-10, v_max=10):
        super(CategoricalCritic, self).__init__()
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta = (v_max - v_min) / (atoms - 1)
        self.z_atoms = torch.linspace(v_min, v_max, atoms)

        self.network = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)),
            ('relu', nn.ReLU()),
            ('w2', nn.Linear(hidden_size, atoms))
        ]))

    def forward(self, state):
        logits = self.network(state)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def get_value(self, state):
        probabilities = self.forward(state)
        return torch.sum(probabilities * self.z_atoms.to(state.device), dim=-1)

    def project_distribution(self, target_distribution, rewards, dones, gamma):
        batch_size = target_distribution.size(0)
        z_atoms = self.z_atoms.view(1, -1).repeat(batch_size, 1).to(target_distribution.device)

        target_atoms = rewards.unsqueeze(1) + gamma * z_atoms * (1 - dones.unsqueeze(1))
        target_atoms = target_atoms.clamp(self.v_min, self.v_max)

        b = (target_atoms - self.v_min) / self.delta
        l = b.floor().long()
        u = b.ceil().long()

        m = torch.zeros_like(target_distribution)
        for i in range(self.atoms):
            m.scatter_add_(1, l.clamp(0, self.atoms - 1), target_distribution[:, i] * (u - b))
            m.scatter_add_(1, u.clamp(0, self.atoms - 1), target_distribution[:, i] * (b - l))

        return m

