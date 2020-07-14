import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std):
        self.lr = 0.0003
        self.betas = (0.9, 0.999)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 80
        
        self.policy = ActorCritic(state_dim, action_dim, action_std)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.policy_old.act(state, memory).data.numpy().flatten()
    
    def train(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(memory.states), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).detach()

        for i in range(self.K_epochs):
            # print('iter', i)
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = - torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = "Pendulum-v0"
    torch.set_num_threads(1)

    max_episodes = 10000  # max training episodes
    max_timesteps = 1500  # max timesteps in one episode
    update_timestep = 4000  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)

    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std)
    time_step = 0

    # training loop
    for epi_iter in range(max_episodes):
        state = env.reset()
        acc_r = 0
        for t in range(max_timesteps):
            time_step += 1
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            if time_step % update_timestep == 0:
                ppo.train(memory)
                memory.clear_memory()
                time_step = 0
            acc_r = acc_r + reward
            # env.render()
            if done:
                break
        print('Episode ', epi_iter, 'reward_sum', acc_r)


    
