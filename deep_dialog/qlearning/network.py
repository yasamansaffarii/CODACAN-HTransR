import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

def init_weight(m, gain=1):
    for name, param in m.named_parameters():
        if name.find('weight') != -1:
            torch.nn.init.xavier_uniform_(param, gain)
        elif name.find('bias') != -1:
            param.data.fill_(0)

class NoisyLinear1(nn.Linear):
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


class NoisyLinear(nn.Linear):
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
        self.epsilon_bias = torch.zeros(self.out_features)

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

class CategoricalDuelNetwork1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, atoms=51):
        super(CategoricalDuelNetwork, self).__init__()
        self.atoms = atoms
        self.output_size = output_size
        self.adv = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, output_size * atoms))]))
        self.vf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, atoms))]))

    def forward(self, inputs, testing=False, log_prob=False):
        v = self.vf(inputs).view(-1, 1, self.atoms)#(16x1 and 276x80)
        adv = self.adv(inputs).view(-1, self.output_size, self.atoms)
        #q = adv
        q = v + adv - adv.mean(1, keepdim=True)
        if log_prob:
            return F.log_softmax(q, -1)
        else:
            return F.softmax(q, -1)


'''class CategoricalDuelNetwork(nn.Module):
    def __init__(self, input_size2, input_size1, input_size, hidden_size, output_size, atoms=51):
        super(CategoricalDuelNetwork, self).__init__()
        self.atoms = atoms
        self.output_size = output_size
        #print('input_size2',input_size2, input_size1, input_size)
        # Ensure input sizes match the actual input dimensions
        self.input2_proj = nn.Linear(input_size2, hidden_size)  # input_size2 should match input2 shape
        self.input1_proj = nn.Linear(input_size1, hidden_size)  # input_size1 should match input1 shape
        self.input_proj = nn.Linear(input_size, hidden_size)    # input_size should match inputs shape

        # Predefined Multihead Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        # Output projection to map attention result back to input_size
        self.attention_output = nn.Linear(hidden_size, input_size)

        self.adv = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, output_size * atoms))]))

        self.vf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, atoms))]))

    def forward(self, input2, input1, inputs, testing=False, log_prob=False):
        # Ensure correct shape for the input before passing through linear layers
        proj_input2 = self.input2_proj(input2)  # Shape: (batch_size, hidden_size)
        proj_input1 = self.input1_proj(input1)  # Shape: (batch_size, hidden_size)
        proj_inputs = self.input_proj(inputs)   # Shape: (batch_size, hidden_size)

        # Add sequence length dimension to proj_inputs
        proj_inputs = proj_inputs.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)

        # Concatenate inputs for attention
        attention_inputs = torch.stack([proj_input2, proj_input1, proj_inputs.squeeze(1)], dim=1)  # Shape: (batch_size, 3, hidden_size)

        # Apply Multihead Attention
        attn_output, _ = self.attention(proj_inputs, attention_inputs, attention_inputs)  # Shape: (batch_size, 1, hidden_size)

        # Map attention output back to input_size
        attended_rep = self.attention_output(attn_output.squeeze(1))  # Shape: (batch_size, input_size)

        v = self.vf(attended_rep).view(-1, 1, self.atoms)
        adv = self.adv(attended_rep).view(-1, self.output_size, self.atoms)
        q = v + adv - adv.mean(1, keepdim=True)
        if log_prob:
            return F.log_softmax(q, -1)
        else:
            return F.softmax(q, -1)'''

'''class CategoricalDuelNetwork(nn.Module):
    def __init__(self, input_size2, input_size1, input_size, hidden_size, output_size, atoms=51):
        super(CategoricalDuelNetwork, self).__init__()
        self.atoms = atoms
        self.output_size = output_size
        
        # Linear projections for inputs
        self.input2_proj = nn.Linear(input_size2, hidden_size)
        self.input1_proj = nn.Linear(input_size1, hidden_size)
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Linear layers for combining attention outputs
        self.attention_output = nn.Linear(hidden_size, input_size)

        # Advantage and value streams
        self.adv = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, output_size * atoms))]))

        self.vf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, atoms))]))

    def forward(self, input2, input1, inputs, testing=False, log_prob=False):
        # Project inputs to hidden space
        proj_input2 = self.input2_proj(input2)  # Shape: (batch_size, hidden_size)
        proj_input1 = self.input1_proj(input1)  # Shape: (batch_size, hidden_size)
        proj_inputs = self.input_proj(inputs)   # Shape: (batch_size, hidden_size)

        # Compute cosine similarity scores
        sim_input1 = F.cosine_similarity(proj_inputs, proj_input1, dim=-1, eps=1e-6)  # Shape: (batch_size,)
        sim_input2 = F.cosine_similarity(proj_inputs, proj_input2, dim=-1, eps=1e-6)  # Shape: (batch_size,)

        # Compute attention weights
        attention_weights = F.softmax(torch.stack([sim_input2, sim_input1, torch.ones_like(sim_input1)], dim=-1), dim=-1)  # Shape: (batch_size, 3)

        # Combine representations based on attention weights
        attention_inputs = torch.stack([proj_input2, proj_input1, proj_inputs], dim=1)  # Shape: (batch_size, 3, hidden_size)
        attended_rep = torch.sum(attention_inputs * attention_weights.unsqueeze(-1), dim=1)  # Shape: (batch_size, hidden_size)

        # Map attended representation back to input_size
        attended_rep = self.attention_output(attended_rep)  # Shape: (batch_size, input_size)

        # Calculate advantage and value streams
        v = self.vf(attended_rep).view(-1, 1, self.atoms)
        adv = self.adv(attended_rep).view(-1, self.output_size, self.atoms)
        q = v + adv - adv.mean(1, keepdim=True)

        if log_prob:
            return F.log_softmax(q, -1)
        else:
            return F.softmax(q, -1)
'''


'''class CategoricalDuelNetwork(nn.Module):
    def __init__(self, input_size2, input_size1, input_size, hidden_size, output_size, atoms=51):
        super(CategoricalDuelNetwork, self).__init__()
        self.atoms = atoms
        self.output_size = output_size
        self.state_element_sizes = [11, 31, 31, 31, 11, 31, 31, 1, 34, 32, 32]
        self.max_size = 34

        # Linear projections for inputs
        self.input2_proj = nn.Linear(11*34, hidden_size)
        self.input1_proj = nn.Linear(11*34, hidden_size)
        self.input_proj = nn.Linear(11*34, hidden_size)

        # Linear layers for combining attention outputs
        self.attention_output = nn.Linear(hidden_size, input_size)

        # Advantage and value streams
        self.adv = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, output_size * atoms))]))

        self.vf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, atoms))]))

    def pad_elements(self, elements, max_size):
        """
        Pad elements to the max size while ensuring padding doesn't affect attention.
        """
        padded = [F.pad(e, (0, max_size - e.shape[-1])) for e in elements]
        return torch.stack(padded, dim=1)  # Shape: (batch_size, num_elements, max_size)

    def forward(self, input2, input1, inputs, testing=False, log_prob=False):

        def process_inputs(inputs):
            # Split inputs by state_element_sizes
            split_inputs = torch.split(inputs, self.state_element_sizes, dim=-1)
            # Pad each element to max_size
            padded_inputs = self.pad_elements(split_inputs, self.max_size)
            # Normalize input vectors to unit length
            normalized_inputs = torch.nn.functional.normalize(padded_inputs, dim=-1)
            # Calculate cosine similarity between all pairs of elements
            attention_weights = torch.matmul(normalized_inputs, normalized_inputs.transpose(-2, -1))
            # Apply attention weights to input embeddings
            attended_inputs = torch.matmul(attention_weights, padded_inputs)
            # Reshape the attended embeddings
            return attended_inputs.view(attended_inputs.shape[0], -1)  # Flatten to shape (batch, 11 * max_size)
        
        # Process each input
        attended_input1 = process_inputs(input1)
        attended_input2 = process_inputs(input2)
        attended_inputs = process_inputs(inputs)

        # Continue with the rest of the forward pass
        proj_input2 = self.input2_proj(attended_input2)  # Shape: (batch_size, hidden_size)
        proj_input1 = self.input1_proj(attended_input1)  # Shape: (batch_size, hidden_size)
        proj_inputs = self.input_proj(attended_inputs)  # Shape: (batch_size, hidden_size)(11x34 and 276x80)

        # Compute cosine similarity scores
        sim_input1 = F.cosine_similarity(proj_inputs, proj_input1, dim=-1, eps=1e-6)  # Shape: (batch_size,)
        sim_input2 = F.cosine_similarity(proj_inputs, proj_input2, dim=-1, eps=1e-6)  # Shape: (batch_size,)

        # Compute attention weights
        attention_weights = F.softmax(torch.stack([sim_input2, sim_input1, torch.ones_like(sim_input1)], dim=-1), dim=-1)  # Shape: (batch_size, 3)

        # Combine representations based on attention weights
        attention_inputs = torch.stack([proj_input2, proj_input1, proj_inputs], dim=1)  # Shape: (batch_size, 3, hidden_size)
        attended_rep = torch.sum(attention_inputs * attention_weights.unsqueeze(-1), dim=1)  # Shape: (batch_size, hidden_size)

        # Map attended representation back to input_size
        attended_rep = self.attention_output(attended_rep)  # Shape: (batch_size, input_size)

        # Calculate advantage and value streams
        v = self.vf(attended_rep).view(-1, 1, self.atoms)
        adv = self.adv(attended_rep).view(-1, self.output_size, self.atoms)
        q = v + adv - adv.mean(1, keepdim=True)

        if log_prob:
            return F.log_softmax(q, -1)
        else:
            return F.softmax(q, -1)'''


class CategoricalDuelNetwork(nn.Module):
    def __init__(self, input_sizes, hidden_size, output_size, atoms=51):
        super(CategoricalDuelNetwork, self).__init__()
        if len(input_sizes) ==3:
            input_size2, input_size1, input_size = input_sizes
        elif len(input_sizes) ==5:
            input_size4 ,input_size3 ,input_size2, input_size1, input_size = input_sizes
        elif len(input_sizes) ==7:
            input_size6 ,input_size5 ,input_size4 ,input_size3 ,input_size2, input_size1, input_size = input_sizes

        self.atoms = atoms
        self.output_size = output_size#movie=43, res=54, tax=41
        #print(self.output_size)
        self.state_element_sizes_res = [11, 31, 31, 31, 11, 31, 31, 1, 34, 32, 32]#276 res
        self.state_len_res = 276
        self.state_element_sizes_tax = [11, 22, 22, 22, 11, 22, 22, 1, 34, 23, 23]#213 taxi
        self.state_len_tax = 213
        self.state_element_sizes_mov = [11, 29, 29, 29, 11, 29, 29, 1, 44, 30, 30]#272 mov
        self.state_len_mov = 272
        self.pad_tax = 374
        self.pad_res = 374
        self.pad_mov = 484
        self.max_size_res = 34
        self.max_size_tax = 34
        self.max_size_mov = 44

        self.state_element_sizes = self.state_element_sizes_mov
        self.state_lens = self.state_len_mov
        self.pad = self.pad_mov
        self.max_size = self.max_size_mov

        # Define W_g for the gate layer
        self.W_g = torch.nn.Parameter(torch.randn(11*self.max_size, 11*self.max_size))  # Adjust dimensions as needed
        # Define W_g1 for the gate2 layer
        self.W_g1 = torch.nn.Parameter(torch.randn(11*self.max_size, 11*self.max_size))  # Adjust dimensions as needed

        # Define a feedforward layer that outputs a tensor of shape [batch_size, 374]
        self.feedforward = torch.nn.Linear(2*11*self.max_size, 11*self.max_size)  # Adjust input size based on concatenation
        # Define a simple feedforward layer to map (batch, 11*self.max_size)to (batch, 276)
        self.feedforward_layer = nn.Linear(11*self.max_size, self.state_lens)

        # Linear projections for inputs
        self.input6_proj = nn.Linear(11*self.max_size, hidden_size)
        self.input5_proj = nn.Linear(11*self.max_size, hidden_size)
        self.input4_proj = nn.Linear(11*self.max_size, hidden_size)
        self.input3_proj = nn.Linear(11*self.max_size, hidden_size)
        self.input2_proj = nn.Linear(11*self.max_size, hidden_size)
        self.input1_proj = nn.Linear(11*self.max_size, hidden_size)
        self.input_proj = nn.Linear(11*self.max_size, hidden_size)

        # Linear layers for combining attention outputs
        self.attention_output = nn.Linear(hidden_size, input_size)

        # Advantage and value streams
        self.adv = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(2*input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, output_size * atoms))]))

        self.vf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(2*input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, atoms))]))

    def pad_elements(self, elements, max_size):
        """
        Pad elements to the max size while ensuring padding doesn't affect attention.
        """
        padded = [F.pad(e, (0, max_size - e.shape[-1])) for e in elements]
        return torch.stack(padded, dim=1)  # Shape: (batch_size, num_elements, max_size)

    def gate_layer(self, S, Z):
        """
        Gate layer that concatenates the attended embedding (Z) with padded inputs (S),
        applies a sigmoid activation on the weighted attended embedding, and returns the concatenated output.
        """
        gated = torch.sigmoid(torch.matmul(Z, self.W_g))  #Apply the gate to the attentional embedding
        # Flatten the padded inputs from shape (batch, num_elements, max_size) to (batch, num_elements * max_size)
        S = S.view(S.shape[0], -1)  # Flatten to shape (batch, num_elements * max_size)            
        return torch.cat([gated, S], dim=-1)  # Concatenate along the feature dimension

    def gate_layer2(self, S, Z):
        """
        Gate layer that concatenates the ff embedding (Z) with prior gate ouput (S),
        applies a sigmoid activation on the weighted ff, and returns the concatenated output.
        """
        gated = torch.sigmoid(torch.matmul(Z, self.W_g))  #Apply the gate to the ff embedding            
        return torch.cat([gated, S], dim=-1)  # Concatenate along the feature dimension

    def split_elementwise_max_pool(self, gated_input):
        """
        Split the input tensor into `num_splits` parts and perform element-wise max pooling.
        """
        # Split the input tensor along the last dimension
        split_inputs = torch.split(gated_input, self.split_size, dim=-1)  # Produces `num_splits` tensors of shape [batch, 374]
        
        # Stack the split tensors along a new dimension
        stacked_splits = torch.stack(split_inputs, dim=0)  # Shape: [num_splits, batch, 374]
        
        # Apply max pooling along the new dimension
        max_pooled_output = torch.max(stacked_splits, dim=0).values  # Shape: [batch, 374]
        
        return max_pooled_output

    def elementwise_max_pool(self, P, G1, G2):

        # Flatten the padded inputs from shape (batch, num_elements, max_size) to (batch, num_elements * max_size)
        P = P.view(P.shape[0], -1)  # Flatten to shape (batch, num_elements * max_size)   

        # Split the G1 tensor along the last dimension
        split_inputs = torch.split(G1, self.pad, dim=-1)  # Produces `num_splits` tensors of shape [batch, 374]
        
        # Select split_inputs[1] for pooling
        G1_part = split_inputs[1]  # Shape: [batch, 374]

        # Split the G2 tensor along the last dimension
        split_inputs2 = torch.split(G2, self.pad, dim=-1)  # Produces `num_splits` tensors of shape [batch, 374]
        
        # Select split_inputs[2] for pooling
        G2_part = split_inputs2[2]  # Shape: [batch, 374]

        # Stack the inputs along a new dimension
        stacked_inputs = torch.stack([P, G1_part, G2_part], dim=0)  # Shape: [3, batch, 374]

        # Perform element-wise max pooling along the first dimension
        pooled_output = torch.max(stacked_inputs, dim=0).values  # Shape: [batch, 374]

        return pooled_output

    def recurrent_att(self, *proj):
        """
        Calculate attended embeddings based on cosine similarity.

        Args:
            proj_input2: Tensor of shape (batch, 276).
            proj_input1: Tensor of shape (batch, 276).
            proj_inputs: Tensor of shape (batch, 276).

        Returns:
            attended_proj_inputs: Tensor of shape (batch, 276), attended embedding for proj_inputs.
        """
        if len(proj) ==3:
            proj_input2, proj_input1, proj_inputs = proj
        elif len(proj) ==5:
            proj_input4 ,proj_input3 ,proj_input2, proj_input1, proj_inputs = proj
        elif len(proj) ==7:
            proj_input6 ,proj_input5 ,proj_input4 ,proj_input3 ,proj_input2, proj_input1, proj_inputs = proj

        # Step 1: Cosine similarity and attended embedding for proj_input1
        # Calculate cosine similarity between proj_input2 and proj_input1
        cosine_sim_1 = torch.nn.functional.cosine_similarity(proj_input2.unsqueeze(1), proj_input1.unsqueeze(1), dim=-1)
        attention_weights_1 = torch.softmax(cosine_sim_1, dim=-1)  # Normalize attention weights
        attended_proj_input1 = torch.matmul(attention_weights_1.unsqueeze(-1), proj_input1.unsqueeze(1)).squeeze(1)  # Apply attention

        # Add and normalize attended_proj_input1
        attended_proj_input1 = torch.nn.functional.normalize(attended_proj_input1 + proj_input1, dim=-1)

        # Step 2: Cosine similarity and attended embedding for proj_inputs
        # Calculate cosine similarity between attended_proj_input1 and proj_inputs
        cosine_sim_2 = torch.nn.functional.cosine_similarity(attended_proj_input1.unsqueeze(1), proj_inputs.unsqueeze(1), dim=-1)
        attention_weights_2 = torch.softmax(cosine_sim_2, dim=-1)  # Normalize attention weights
        attended_proj_inputs = torch.matmul(attention_weights_2.unsqueeze(-1), proj_inputs.unsqueeze(1)).squeeze(1)  # Apply attention

        # Add and normalize attended_proj_inputs
        attended_proj_inputs = torch.nn.functional.normalize(attended_proj_inputs + proj_inputs, dim=-1)
        # Pass through the feedforward layer(374->276)
        mapped_proj_inputs = self.feedforward_layer(attended_proj_inputs)

        return mapped_proj_inputs


    def forward(self, *inp, testing=False, log_prob=False):
        #print("eeeeeeeeeeeeee",len(inp))
        if len(inp) == 3:
            input1, input2, inputs = inp
        elif len(inp) ==5:
            input1, input2, input3, input4, inputs = inp
        elif len(inp) ==7:
            input1, input2, input3, input4, input5, input6, inputs = inp


        def process_inputs(inputs):
            # Split inputs by state_element_sizes
            split_inputs = torch.split(inputs, self.state_element_sizes, dim=-1)
            # Pad each element to max_size
            padded_inputs = self.pad_elements(split_inputs, self.max_size)
            # Normalize input vectors to unit length
            normalized_inputs = torch.nn.functional.normalize(padded_inputs, dim=-1)
            # Calculate cosine similarity between all pairs of elements
            attention_weights = torch.matmul(normalized_inputs, normalized_inputs.transpose(-2, -1))
            # Apply attention weights to input embeddings
            attended_inputs = torch.matmul(attention_weights, padded_inputs)
            # Reshape the attended embeddings
            return attended_inputs.view(attended_inputs.shape[0], -1),padded_inputs  # Flatten to shape (batch, 11 * max_size)
        
        # Process each input
        if len(inp) ==3:
            attended_input1 , padded_inputs1 = process_inputs(input1)
            attended_input2 , padded_inputs2 = process_inputs(input2)
            attended_inputs , padded_inputs = process_inputs(inputs)
        elif len(inp) ==5:
            attended_input1 , padded_inputs1 = process_inputs(input1)
            attended_input2 , padded_inputs2 = process_inputs(input2)
            attended_input3 , padded_inputs3 = process_inputs(input3)
            attended_input4 , padded_inputs4 = process_inputs(input4)
            attended_inputs , padded_inputs = process_inputs(inputs)
        elif len(inp) ==7:
            attended_input1 , padded_inputs1 = process_inputs(input1)
            attended_input2 , padded_inputs2 = process_inputs(input2)
            attended_input3 , padded_inputs3 = process_inputs(input3)
            attended_input4 , padded_inputs4 = process_inputs(input4)
            attended_input5 , padded_inputs5 = process_inputs(input5)
            attended_input6 , padded_inputs6 = process_inputs(input6)
            attended_inputs , padded_inputs = process_inputs(inputs)

        # Apply the gate mechanism
        if len(inp) ==3:
            gated_input1 = self.gate_layer(padded_inputs1, attended_input1)
            gated_input2 = self.gate_layer(padded_inputs2, attended_input2)
            gated_inputs = self.gate_layer(padded_inputs, attended_inputs)
        elif len(inp) ==5:
            gated_input1 = self.gate_layer(padded_inputs1, attended_input1)
            gated_input2 = self.gate_layer(padded_inputs2, attended_input2)
            gated_input3 = self.gate_layer(padded_inputs3, attended_input3)
            gated_input4 = self.gate_layer(padded_inputs4, attended_input4)
            gated_inputs = self.gate_layer(padded_inputs, attended_inputs)
        elif len(inp) ==7:
            gated_input1 = self.gate_layer(padded_inputs1, attended_input1)
            gated_input2 = self.gate_layer(padded_inputs2, attended_input2)
            gated_input3 = self.gate_layer(padded_inputs3, attended_input3)
            gated_input4 = self.gate_layer(padded_inputs4, attended_input4)
            gated_input5 = self.gate_layer(padded_inputs5, attended_input5)
            gated_input6 = self.gate_layer(padded_inputs6, attended_input6)
            gated_inputs = self.gate_layer(padded_inputs, attended_inputs)


        #print(gated_input1.shape)#torch.Size([1, 748])


        # Apply the feedforward layer on the gated input
        if len(inp) == 3:
            output1 = self.feedforward(gated_input1)
            output2 = self.feedforward(gated_input2)
            output = self.feedforward(gated_inputs)
        elif len(inp) == 5:
            output1 = self.feedforward(gated_input1)
            output2 = self.feedforward(gated_input2)
            output3 = self.feedforward(gated_input3)
            output4 = self.feedforward(gated_input4)
            output = self.feedforward(gated_inputs)
        elif len(inp) == 7:
            output1 = self.feedforward(gated_input1)
            output2 = self.feedforward(gated_input2)
            output3 = self.feedforward(gated_input3)
            output4 = self.feedforward(gated_input4)
            output5 = self.feedforward(gated_input5)
            output6 = self.feedforward(gated_input6)    
            output = self.feedforward(gated_inputs)


        # Apply the gate mechanism
        if len(inp) == 3:
            gated_input11 = self.gate_layer(gated_input1, output1)
            gated_input22 = self.gate_layer(gated_input2, output2)
            gated_inputss = self.gate_layer(gated_inputs, output)
        elif len(inp) == 5:
            gated_input11 = self.gate_layer(gated_input1, output1)
            gated_input22 = self.gate_layer(gated_input2, output2)
            gated_input33 = self.gate_layer(gated_input3, output3)
            gated_input44 = self.gate_layer(gated_input4, output4)            
            gated_inputss = self.gate_layer(gated_inputs, output)

        elif len(inp) == 7:
            gated_input11 = self.gate_layer(gated_input1, output1)
            gated_input22 = self.gate_layer(gated_input2, output2)
            gated_input33 = self.gate_layer(gated_input3, output3)
            gated_input44 = self.gate_layer(gated_input4, output4) 
            gated_input55 = self.gate_layer(gated_input5, output5)
            gated_input66 = self.gate_layer(gated_input6, output6)
            gated_inputss = self.gate_layer(gated_inputs, output)

      
        # Perform split_element-wise max pooling for each gated input
        #pooled_output11 = self.split_elementwise_max_pool(gated_input11)
        #pooled_output22 = self.split_elementwise_max_pool(gated_input22)
        #pooled_outputss = self.split_elementwise_max_pool(gated_inputss)

        # Perform element-wise max pooling for each gated input#batch*374
        if len(inp) ==3:
            pooled_output11 = self.elementwise_max_pool(padded_inputs1, gated_input1, gated_input11)
            pooled_output22 = self.elementwise_max_pool(padded_inputs2, gated_input2, gated_input22)
            pooled_outputss = self.elementwise_max_pool(padded_inputs, gated_inputs, gated_inputss)
        elif len(inp) == 5:
            pooled_output11 = self.elementwise_max_pool(padded_inputs1, gated_input1, gated_input11)
            pooled_output22 = self.elementwise_max_pool(padded_inputs2, gated_input2, gated_input22)
            pooled_output33 = self.elementwise_max_pool(padded_inputs3, gated_input3, gated_input33)
            pooled_output44 = self.elementwise_max_pool(padded_inputs4, gated_input4, gated_input44)
            pooled_outputss = self.elementwise_max_pool(padded_inputs, gated_inputs, gated_inputss)          
        elif len(inp) == 7:
            pooled_output11 = self.elementwise_max_pool(padded_inputs1, gated_input1, gated_input11)
            pooled_output22 = self.elementwise_max_pool(padded_inputs2, gated_input2, gated_input22)
            pooled_output33 = self.elementwise_max_pool(padded_inputs3, gated_input3, gated_input33)
            pooled_output44 = self.elementwise_max_pool(padded_inputs4, gated_input4, gated_input44)
            pooled_output55 = self.elementwise_max_pool(padded_inputs5, gated_input5, gated_input55)
            pooled_output66 = self.elementwise_max_pool(padded_inputs6, gated_input6, gated_input66)            
            pooled_outputss = self.elementwise_max_pool(padded_inputs, gated_inputs, gated_inputss)

        if len(inp) == 3:
             hier1= self.recurrent_att(pooled_output11, pooled_output22, pooled_outputss)#batch*374
        elif len(inp) == 5:
             hier1= self.recurrent_att(pooled_output11, pooled_output22, pooled_output33, pooled_output44, pooled_outputss)#batch*374
        elif len(inp) == 7:  
             hier1= self.recurrent_att(pooled_output11, pooled_output22, pooled_output33, pooled_output44, pooled_output55, pooled_output66, pooled_outputss)#batch*374
       



        # Continue with the rest of the forward pass
        if len(inp) == 3:
            proj_input2 = self.input2_proj(pooled_output22)  # Shape: (batch_size, hidden_size)
            proj_input1 = self.input1_proj(pooled_output11)  # Shape: (batch_size, hidden_size)
            proj_inputs = self.input_proj(pooled_outputss)  # Shape: (batch_size, hidden_size)
        elif len(inp) == 5:
            proj_input4 = self.input4_proj(pooled_output44)  # Shape: (batch_size, hidden_size)
            proj_input3 = self.input3_proj(pooled_output33)  # Shape: (batch_size, hidden_size)
            proj_input2 = self.input2_proj(pooled_output22)  # Shape: (batch_size, hidden_size)
            proj_input1 = self.input1_proj(pooled_output11)  # Shape: (batch_size, hidden_size)
            proj_inputs = self.input_proj(pooled_outputss)  # Shape: (batch_size, hidden_size)          
        elif len(inp) == 7:

            proj_input6 = self.input6_proj(pooled_output66)  # Shape: (batch_size, hidden_size)
            proj_input5 = self.input5_proj(pooled_output55)  # Shape: (batch_size, hidden_size)
            proj_input4 = self.input4_proj(pooled_output44)  # Shape: (batch_size, hidden_size)
            proj_input3 = self.input3_proj(pooled_output33)  # Shape: (batch_size, hidden_size)
            proj_input2 = self.input2_proj(pooled_output22)  # Shape: (batch_size, hidden_size)
            proj_input1 = self.input1_proj(pooled_output11)  # Shape: (batch_size, hidden_size)
            proj_inputs = self.input_proj(pooled_outputss)  # Shape: (batch_size, hidden_size)          


        # Compute cosine similarity scores
        if len(inp) == 3:
            sim_input1 = F.cosine_similarity(proj_inputs, proj_input1, dim=-1, eps=1e-6)  # Shape: (batch_size,)
            sim_input2 = F.cosine_similarity(proj_inputs, proj_input2, dim=-1, eps=1e-6)  # Shape: (batch_size,)

        elif len(inp) == 5:
            sim_input1 = F.cosine_similarity(proj_inputs, proj_input1, dim=-1, eps=1e-6)  # Shape: (batch_size,)
            sim_input2 = F.cosine_similarity(proj_inputs, proj_input2, dim=-1, eps=1e-6)  # Shape: (batch_size,)
            sim_input3 = F.cosine_similarity(proj_inputs, proj_input3, dim=-1, eps=1e-6)  # Shape: (batch_size,)
            sim_input4 = F.cosine_similarity(proj_inputs, proj_input4, dim=-1, eps=1e-6)  # Shape: (batch_size,)

        elif len(inp) == 7:
            sim_input1 = F.cosine_similarity(proj_inputs, proj_input1, dim=-1, eps=1e-6)  # Shape: (batch_size,)
            sim_input2 = F.cosine_similarity(proj_inputs, proj_input2, dim=-1, eps=1e-6)  # Shape: (batch_size,)
            sim_input3 = F.cosine_similarity(proj_inputs, proj_input3, dim=-1, eps=1e-6)  # Shape: (batch_size,)
            sim_input4 = F.cosine_similarity(proj_inputs, proj_input4, dim=-1, eps=1e-6)  # Shape: (batch_size,)
            sim_input5 = F.cosine_similarity(proj_inputs, proj_input5, dim=-1, eps=1e-6)  # Shape: (batch_size,)
            sim_input6 = F.cosine_similarity(proj_inputs, proj_input6, dim=-1, eps=1e-6)  # Shape: (batch_size,)


        # Compute attention weights
        if len(inp) == 3:
            attention_weights = F.softmax(torch.stack([sim_input2, sim_input1, torch.ones_like(sim_input1)], dim=-1), dim=-1)  # Shape: (batch_size, 3)
        elif len(inp) == 5:
            # Stack all similarities and bias for attention
            similarities = torch.stack([sim_input1, sim_input2, sim_input3, sim_input4, torch.ones_like(sim_input1)], dim=-1)  # Shape: (batch_size, 7)
            # Apply softmax to compute attention weights
            attention_weights = F.softmax(similarities, dim=-1)  # Shape: (batch_size, 7)
        elif len(inp) == 7:
            # Stack all similarities and bias for attention
            similarities = torch.stack([sim_input1, sim_input2, sim_input3, sim_input4, sim_input5, sim_input6, torch.ones_like(sim_input1)], dim=-1)  # Shape: (batch_size, 7)
            # Apply softmax to compute attention weights
            attention_weights = F.softmax(similarities, dim=-1)  # Shape: (batch_size, 7)

        # Combine representations based on attention weights
        if len(inp) ==3:
            attention_inputs = torch.stack([proj_input2, proj_input1, proj_inputs], dim=1)  # Shape: (batch_size, 3, hidden_size)
            attended_rep = torch.sum(attention_inputs * attention_weights.unsqueeze(-1), dim=1)  # Shape: (batch_size, hidden_size)
        elif len(inp) == 5:
            attention_inputs = torch.stack([proj_input4, proj_input3, proj_input2, proj_input1, proj_inputs], dim=1)  # Shape: (batch_size, 3, hidden_size)
            attended_rep = torch.sum(attention_inputs * attention_weights.unsqueeze(-1), dim=1)  # Shape: (batch_size, hidden_size)
        elif len(inp) == 7:
            attention_inputs = torch.stack([proj_input6, proj_input5, proj_input4, proj_input3, proj_input2, proj_input1, proj_inputs], dim=1)  # Shape: (batch_size, 3, hidden_size)
            attended_rep = torch.sum(attention_inputs * attention_weights.unsqueeze(-1), dim=1)  # Shape: (batch_size, hidden_size)
        else:
            # Handle unexpected input lengths
            print(len(inp))
            raise ValueError(f"Unexpected input length: {len(inp)}. Expected lengths are 3, 5, or 7.")


        # Map attended representation back to input_size
        attended_rep = self.attention_output(attended_rep)  # Shape: (batch_size, input_size)(batch*276)
     
        # Add and normalize attended_proj_inputs
        hier2 = torch.nn.functional.normalize(attended_rep + inputs, dim=-1)#batch*276

        # Element-wise max pooling
        max_pooled = torch.max(hier1, hier2)

        # Concatenate tensors along the last dimension (dim=1)
        concatenated = torch.cat((inputs, max_pooled), dim=1)#batch,2*276

        # Calculate advantage and value streams
        v = self.vf(concatenated).view(-1, 1, self.atoms)
        adv = self.adv(concatenated).view(-1, self.output_size, self.atoms)
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

