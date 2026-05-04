import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

class CDFC_Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CDFC_Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(64, output_dim)
        self.log_std_layer = nn.Linear(64, output_dim)

    def custom_sigmoid(self, upper_lim, lower_lim, x, c=1.2):
        x = ((upper_lim - lower_lim) / (1 + torch.exp(-c * x))) + lower_lim
        return x

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_layer(x)
        # set limits for the custom sigmoid
        upper_lim = torch.full_like(mean, 80)
        lower_lim = torch.full_like(mean, 0)
        # y elements is 5 km under the border
        upper_lim[1::2] = 65
        mean = self.custom_sigmoid(upper_lim, lower_lim, mean)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-7, max=2)
        std = torch.exp(log_std)
        return mean, std


class CFC_Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CFC_Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(64, output_dim)
        self.log_std_layer = nn.Linear(64, output_dim)

    def custom_sigmoid(self, upper_lim, lower_lim, x, c=1.0):
        x = ((upper_lim - lower_lim) / (1 + torch.exp(-c * x))) + lower_lim
        return x

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_layer(x)
        # Define bounds
        upper_lim = torch.full_like(mean, 80)
        lower_lim = torch.full_like(mean, 0)
        pattern = torch.tensor([80, 65, 155], device=mean.device)
        upper_lim = pattern.repeat(mean.shape[-1] // 3)
        # angle lower limit is 25 degrees
        lower_lim[2::3] = 25
        mean = self.custom_sigmoid(upper_lim, lower_lim, mean)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-7, max=2)
        std = torch.exp(log_std)
        return mean, std


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)


class PPO:
    def __init__(self, actor, critic, actor_lr=2.5e-4, critic_lr=1e-3, gamma=0.99, clip_eps=0.2):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, state_tensor, episode, monitor, i, j):
        mean, std = self.actor(state_tensor)
        normal_dist = dist.Normal(mean, std)
        action = normal_dist.rsample()
        action = action.view(i, j)
        action = action.detach().cpu().numpy()
        value = self.critic(state_tensor).detach().cpu().numpy()
        monitor.append({
            'episode': episode,
            'mean': mean.detach().cpu().numpy().tolist(),
            'std': std.detach().cpu().numpy().tolist(),
            'state': action.tolist()
        })
        return action, value, monitor

    def train(self, state_tensor, next_state_tensor, reward, value, next_value):
        advantage = reward + self.gamma * next_value - value
        mean, std = self.actor(state_tensor.flatten())
        dist_current = dist.Normal(mean, std)
        predicted_positions = dist_current.sample()
        old_log_probs = dist_current.log_prob(predicted_positions.flatten()).sum(dim=-1).detach()

        for _ in range(5):
            mean_new, std_new = self.actor(next_state_tensor.flatten())
            dist_new = dist.Normal(mean_new, std_new)
            new_predict_pos = dist_new.sample()
            new_log_probs = dist_new.log_prob(new_predict_pos.flatten()).sum(dim=-1).detach()
            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(policy_ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            advantage_tensor = torch.tensor(advantage, requires_grad=True).to(self.device)
            policy_loss = -torch.min(policy_ratio * advantage_tensor, clipped_ratio * advantage_tensor)

            current_value_prediction = self.critic(state_tensor.flatten())
            target_value = reward + self.gamma * next_value

            target_value_tensor = torch.tensor(target_value, dtype=torch.float32).to(self.device).detach()

            value_loss = (current_value_prediction - target_value_tensor) ** 2

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()