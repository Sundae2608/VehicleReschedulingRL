from .policy import Policy
from .replay_buffer import ReplayBuffer

import copy
import os
import pickle
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class CommNetworkParams:
    def __init__(self, hidden_size, k, buffer_size, batch_size, update_every, learning_rate, tau, gamma, buffer_min_size,
                 use_gpu):
        self.hidden_size = hidden_size
        self.k = k
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.buffer_min_size = buffer_min_size
        self.use_gpu = use_gpu


class CommNetwork(nn.Module):
    """CommNet (https://arxiv.org/pdf/1605.07736.pdf)"""

    def __init__(self,
                 state_size,
                 action_size,
                 k,
                 hidden_size):
        """
        Initialize the CommNetwork.
        :param state_size: Size of the input to the network
        :param action_size: Number of actions that each network produce
        :param k: The number of layers in the network
        :param hidden_size: The size of the hidden state
        """

        # Encoding layer
        self.encoder = nn.Linear(state_size, hidden_size)

        # Create k modules
        self.f = []
        for i in range(k):
            self.f.append(nn.Linear(2 * hidden_size, hidden_size))

        # Action layer
        self.action_layer = nn.Linear(2 * hidden_size, action_size)

    def forward(self, x: torch.Tensor):
        """
        Feed forward to get the action of vector of each agent
        :param x: An n-by-s tensor of states, representing the state of all agents in the network.
            n is the number of agents
            s is the size of the state
        :return: An n-by-a tensor of states, representing the action vectors of all agents in the network
            n is the number of agents
            a is the number of available actions
        """
        # Get the number of agents
        num_agents = x.shape[0]

        # Encode the first state
        h_arr = F.relu(self.encoder(x))

        # For each of the communication module, produce hidden value for each agents.
        for f in self.f:
            h_mean = torch.mean(h_arr, 1)
            c_arr = h_mean.repeat((num_agents, 1))
            hc_arr = torch.cat((h_arr, c_arr), 1)
            h_arr = F.tanh(f(hc_arr))

        # Producer actions
        h_mean = torch.mean(h_arr, 1)
        c_arr = h_mean.repeat((num_agents, 1))
        hc_arr = torch.cat((h_arr, c_arr), 1)
        return self.action_layer(hc_arr)


class CommNetworkPolicy(Policy):
    """CommNetwork policy"""

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 parameters: CommNetworkParams,
                 evaluation_mode: bool = False):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = True
        self.hidden_size = 1

        self.hidden_size = parameters.hidden_size
        self.k = parameters.k
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size

        # Device
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # CommNet
        self.commnet_local = CommNetwork(self.state_size, self.action_size, self.k, self.hidden_size)

        if not evaluation_mode:
            self.commnet_target = copy.deepcopy(self.commnet_local)
            self.optimizer = optim.Adam(self.commnet_local.parameters(), lr=self.learning_rate)
            self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)

            self.t_step = 0
            self.loss = 0.0

    def act(self, states, eps=0.):
        """
        Produce action vectors for all the agent states
        :param states: A 2d n-by-s array representing the states of all agents.
            - n is the number of agent
            - s is the number of states
        :param eps: Exploration coefficient, where a random actions would be conducted.
        :return: An n-by-1 vector of actions
        """
        n = states.shape[0]
        states = torch.from_numpy(states).float().to(self.device)
        self.commnet_local.eval()
        with torch.no_grad():
            action_values = self.commnet_local(states)
        self.commnet_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            return np.random.choice(np.arange(self.action_size), size=n)

    def step(self, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local models
        q_expected = self.commnet_local(states).gather(1, actions)

        if self.double_dqn:
            # Double DQN
            q_best_action = self.commnet_local(next_states).max(1)[1]
            q_targets_next = self.commnet_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            # DQN
            q_targets_next = self.commnet_target(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute loss
        self.loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.commnet_local, self.commnet_target, self.tau)

    def _soft_update(self, local_model, target_model, tau):
        # Soft update models parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.commnet_local.state_dict(), filename + ".local")
        torch.save(self.commnet_target.state_dict(), filename + ".target")

    def load(self, filename):
        if os.path.exists(filename + ".local"):
            self.commnet_local.load_state_dict(torch.load(filename + ".local"))
        if os.path.exists(filename + ".target"):
            self.commnet_target.load_state_dict(torch.load(filename + ".target"))

    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)

    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)

    def test(self):
        self.act(np.array([[0] * self.state_size]))
        self._learn()

