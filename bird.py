import os
import warnings

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import random
from collections import deque
import matplotlib.pyplot as plt

class EnvProcessor(gym.Wrapper):
    def __init__(self, env):
        super(EnvProcessor, self).__init__(env)
        self.width = 84
        self.height = 84
        self.frame_stack_len = 4    
        self.frames = deque(maxlen=self.frame_stack_len)
        self.skip = 2 
        self.alignment_scale = 0.05
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.frame_stack_len, self.height, self.width),
            dtype=np.uint8
        )

    def step(self, action):
        total_reward = 0
        
        for _ in range(self.skip):
            _, reward, terminated, truncated, info = self.env.step(action)
            
            if reward == 1:
                reward = 1.0
            
            if terminated:
                reward = -1.0 
                
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        frame = self.env.render()
        processed_frame = self._process_frame(frame)
        self.frames.append(processed_frame)
        
        return self._get_stacked_obs(), total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        frame = self.env.render()
        processed_frame = self._process_frame(frame)
        self.frames.clear()
        for _ in range(self.frame_stack_len):
            self.frames.append(processed_frame)
        return self._get_stacked_obs(), info

    def _process_frame(self, frame):
        if frame is None:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized

    def _get_stacked_obs(self):
        return np.stack(self.frames, axis=0)

class DDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )

        conv_out = self._get_conv_out(input_shape)

        self.value = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        value = self.value(x)
        advantage = self.advantage(x)

        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Training on: {self.device} (Model: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99997
        self.learning_rate = 1e-4
        self.target_update_freq = 6000
        self.memory_size = 400000
        self.start_training_step = 3000 

        input_shape = env.observation_space.shape
        num_actions = env.action_space.n
        
        self.policy_net = DDQN(input_shape, num_actions).to(self.device)
        self.target_net = DDQN(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(self.memory_size)
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_t = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device) / 255.0
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < self.start_training_step:
            return None

    
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device) / 255.0
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device) / 255.0
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():

            best_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)

            next_q_values = self.target_net(next_states).gather(1, best_actions)
            
            target_q = rewards + (1 - dones) * self.gamma * next_q_values
        

        current_q = self.policy_net(states).gather(1, actions)
        
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()

        return loss.item()


    def save(self, path="flappy_checkpoint.pth"):
        checkpoint = {
            'model_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path} (Epsilon: {self.epsilon:.3f})")

    def load(self, path="flappy_checkpoint.pth"):
        if not os.path.exists(path):
            print("No checkpoint found. Starting from scratch.")
            return

        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['model_state'])
        self.target_net.load_state_dict(checkpoint['target_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        
        self.steps = checkpoint.get('steps', self.steps)
        print(f"Loaded checkpoint! Resuming with Epsilon: {self.epsilon:.3f}")

    def start(self, num_episodes=60000):
        scores = []
        recent_scores = deque(maxlen=100)

        for episode in range(num_episodes):
            state, info = self.env.reset()
            total_reward = 0
            
            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.steps += 1

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                if self.steps > self.start_training_step and  self.steps % 2 == 0:
                    _loss = self.learn()

                if self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                if done:
                    break

            scores.append(total_reward)
            recent_scores.append(total_reward)


            if episode % 100 == 0:

                avg_score = np.mean(recent_scores) if recent_scores else 0
                print(f"Episode {episode}, Avg Score (Last 100): {avg_score:.2f}, Epsilon: {self.epsilon:.2f}, Memory: {len(self.memory)}")

            if episode % 1000 == 0:
                self.save()

        return scores

if __name__ == "__main__":

    print("Initializing Environment...")
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    

    env = EnvProcessor(env)
    
    agent = Agent(env)
    
    if os.path.exists("flappy_checkpoint.pth"):
        agent.load("flappy_checkpoint.pth")
    else:
        print("Starting training from scratch...")

    scores = agent.start()
    
    env.close()
    
    plt.plot(scores)
    plt.title("Flappy Bird Training Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()
