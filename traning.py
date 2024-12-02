import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import copy
import numpy as np
import gymnasium as gym
import cv2
from ultralytics import YOLO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("your yolo OD path").to(device)
print(device)

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, log_std=0):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head_steering = nn.Linear(net_width, action_dim)
		self.mu_head_steering.weight.data.mul_(0.1)
		self.mu_head_steering.bias.data.mul_(0.0)

		self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

	def forward(self, state):
		if state.dim() == 1:
			state = state.unsqueeze(0)
		
		a = torch.relu(self.l1(state))
		a = torch.relu(self.l2(a))
		mu = torch.tanh(self.mu_head_steering(a))
		return mu

	def get_dist(self, state):
		mu = self.forward(state)
		action_log_std = self.action_log_std.expand(mu.size(0), -1)
		action_std = torch.exp(action_log_std)

		dist = Normal(mu, action_std)
		return dist

	def deterministic_act(self, state):
		return self.forward(state)

class Critic(nn.Module):
	def __init__(self, state_dim,net_width):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.tanh(self.C1(state))
		v = torch.tanh(self.C2(v))
		v = self.C3(v)
		return v

class PPO_agent():
	def __init__(self):
	
		self.state_dim = 73
		self.action_dim = 1
		self.net_width = 150
		self.T_horizon = 2048
		
		# Training parameters
		self.gamma = 0.99
		self.lambd = 0.95
		self.clip_rate = 0.2
		self.K_epochs = 10
		self.a_optim_batch_size = 64
		self.c_optim_batch_size = 64
		self.entropy_coef = 1e-3
		self.entropy_coef_decay = 0.99
		self.l2_reg = 1e-3

		# Choose distribution for the actor
		self.actor = Actor(self.state_dim, self.action_dim, self.net_width).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=2e-4)

		# Build Critic
		self.critic = Critic(self.state_dim, self.net_width).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-4)

		# Build Trajectory holder
		self.s_hoder = torch.zeros((self.T_horizon, self.state_dim), dtype=torch.float32).to(device)
		self.a_hoder = torch.zeros((self.T_horizon, self.action_dim), dtype=torch.float32).to(device)
		self.r_hoder = torch.zeros((self.T_horizon, 1), dtype=torch.float32).to(device)
		self.s_next_hoder = torch.zeros((self.T_horizon, self.state_dim), dtype=torch.float32).to(device)
		self.logprob_a_hoder = torch.zeros((self.T_horizon, self.action_dim), dtype=torch.float32).to(device)
		self.done_hoder = torch.zeros((self.T_horizon, 1), dtype=torch.float32).to(device)
		self.dw_hoder = torch.zeros((self.T_horizon, 1), dtype=torch.float32).to(device)

	def put_data(self, now_state, action, reward, next_state, logprob_a, done, dw, idx):
		# Convert numpy arrays to torch tensors and move to correct device
		if isinstance(now_state, np.ndarray):
			self.s_hoder[idx] = torch.from_numpy(now_state).float().to(device)
		else:
			self.s_hoder[idx] = now_state.float().to(device)
		
		if isinstance(action, np.ndarray):
			self.a_hoder[idx] = torch.from_numpy(action).float().to(device)
		else:
			self.a_hoder[idx] = action.float().to(device)
		
		if isinstance(next_state, np.ndarray):
			self.s_next_hoder[idx] = torch.from_numpy(next_state).float().to(device)
		else:
			self.s_next_hoder[idx] = next_state.float().to(device)
		
		if isinstance(logprob_a, np.ndarray):
			self.logprob_a_hoder[idx] = torch.from_numpy(logprob_a).float().to(device)
		else:
			self.logprob_a_hoder[idx] = logprob_a.float().to(device)
		
		# Handle scalar values
		self.r_hoder[idx] = torch.tensor([reward], dtype=torch.float32).to(device)
		self.done_hoder[idx] = torch.tensor([done], dtype=torch.float32).to(device)
		self.dw_hoder[idx] = torch.tensor([dw], dtype=torch.float32).to(device)

	def select_action(self, state, deterministic):
		with torch.no_grad():
			if deterministic:
				# only used when evaluate the policy
				a = self.actor.deterministic_act(state)
				return a.cpu().numpy()[0], None  # action is in shape (adim, 0)
			else:
				# only used when interact with the env
				dist = self.actor.get_dist(state)
				a = dist.sample()
				a = torch.clamp(a, -1, 1)
				logprob_a = dist.log_prob(a).cpu().numpy().flatten()
				return a.cpu().numpy()[0], logprob_a  # both are in shape (adim, 0)
	def train(self):
		self.entropy_coef*=self.entropy_coef_decay

		'''Prepare PyTorch data from Numpy data'''
		s = self.s_hoder
		a = self.a_hoder
		r = self.r_hoder
		s_next = self.s_next_hoder
		logprob_a = self.logprob_a_hoder
		done = self.done_hoder
		dw = self.dw_hoder

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(s)
			vs_ = self.critic(s_next)

			'''dw for TD_target and Adv'''
			deltas = r + self.gamma * vs_ * (1-dw) - vs
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (1-mask)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(device)
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps


		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
		for i in range(self.K_epochs):

			#Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(device)
			s, a, td_target, adv, logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			'''update the actor'''
			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
				distribution = self.actor.get_dist(s[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(a[index])
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
				c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

def distances_and_speed_tensor(now_results, now_speed):
    tensor = torch.tensor(now_results, dtype=torch.float32).to(device).view(-1)
    tensor = torch.cat((tensor, torch.tensor([now_speed], dtype=torch.float32).to(device)))
    return tensor

def apply_green_filter(image):
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define the range for green color
    lower_green = np.array([40, 40, 40])  
    upper_green = np.array([80, 255, 255])

    # Create a mask for green areas
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    return green_mask

def detector(next_state):
    # Convert NumPy image (state) to BGR format expected by OpenCV/YOLO
    next_state_bgr = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
    
    # Apply green filter to detect green areas
    green_mask = apply_green_filter(next_state)

    # Run inference on the image
    results = model(next_state_bgr, conf=0.5)
    result = results[0]
    boxes = result.boxes

    touch = False

    if len(boxes) != 0:

        # Draw detected boxes and check for green contact
        corners = []
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert bounding box to integers

            # Get corners of the bounding box
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]  # (top-left, top-right, bottom-left, bottom-right)

            # Check if any corner touches the green area
            for corner in corners:
                if green_mask[corner[1], corner[0]] == 255:  # If green pixel at this corner
                    touch = True
                    break
                
        car_center_x = (x1 + x2) / 2
        car_center_y = (y1 + y2) / 2

        # Modified ray-casting code
        angles = np.arange(0, 360, 10)
        results = []
        for angle in angles:
            found_green = False
            rad = np.deg2rad(angle)

            for distance in range(1, 20 + 1):
                x = int(car_center_x + distance * np.cos(rad))
                y = int(car_center_y + distance * np.sin(rad))

                if green_mask[y, x] > 0:  # Check if the pixel is green
                    # Calculate x and y distances from car center
                    dx = x - car_center_x
                    dy = y - car_center_y
                    results.append((dx, dy))
                    found_green = True
                    break
                
            if not found_green:
                # If no green found, store maximum distance in that direction
                max_dx = 20 * np.cos(rad)
                max_dy = 20 * np.sin(rad)
                results.append((max_dx, max_dy))

        return touch, results
    else:
        return "none", "none"

def save_model(agent, best_reward):
	env_test = gym.make("CarRacing-v3")
	env_seed = 0
	now_speed = 0
	first_action = np.array([0, 0, 0])
	action_tensor = np.array([0, 0.1, 0])
	
	traj_lenth, total_steps, total_train = 0, 0, 0
	for i in range(1):
		run_reward = 0
		done = False
		now_state = env_test.reset(seed=env_seed)[0]
		
		while not done:
			break_value = 0
			if now_speed > 40:
				break_value = 1
			now_touch, now_results = detector(now_state)
			if now_results != "none":
				now_results_and_speed = distances_and_speed_tensor(now_results, now_speed)
				action, logprob_a = agent.select_action(now_results_and_speed, deterministic=True)
				action_tensor[0] = action
				action_tensor[2] = break_value

				next_state, reward, done, next_speed, info = env_test.step(action_tensor)
				total_steps += 1  # Increment total steps
				
				next_touch, next_results = detector(next_state)

				if next_touch == True:
					done = True

				run_reward += reward
				
				if next_results != "none":
					reward += next_speed

					now_state = next_state
					now_speed = next_speed 
				else:
					done = True
			else:
				next_state, reward, done, next_speed, info = env_test.step(first_action)
				now_state = next_state
				now_speed = next_speed
			print(f"Test total reward: {run_reward}")
	env_test.close()
	input()
	if run_reward > best_reward:
		torch.save(agent.actor.state_dict(), f"actor-{int(run_reward)}.pt")
		return int(run_reward)
	else:
		return int(best_reward)
				
			
def main():
	env = gym.make("CarRacing-v3")
	agent = PPO_agent()

	env_seed = 0
	now_speed = 0
	first_action = np.array([0, 0, 0])
	action_tensor = np.array([0, 0.1, 0])
	best_reward = 100
	
	traj_lenth, total_steps, total_train = 0, 0, 0
	while total_train < 50:
		run_reward = 0
		done = False
		now_state = env.reset(seed=env_seed)[0]
		
		while not done:
			break_value = 0
			if now_speed > 40:
				break_value = 1
			now_touch, now_results = detector(now_state)
			if now_results != "none":
				now_results_and_speed = distances_and_speed_tensor(now_results, now_speed)
				action, logprob_a = agent.select_action(now_results_and_speed, deterministic=False)
				action_tensor[0] = action
				action_tensor[2] = break_value

				next_state, reward, done, next_speed, info = env.step(action_tensor)
				total_steps += 1  # Increment total steps
				
				next_touch, next_results = detector(next_state)

				if next_touch == True:
					done = True
					agent.put_data(now_results_and_speed, action, reward, next_results_and_speed, logprob_a, done, 0, traj_lenth)
					traj_lenth += 1

				run_reward += reward
				
				if next_results != "none":
					reward += next_speed
					next_results_and_speed = distances_and_speed_tensor(next_results, next_speed)
					agent.put_data(now_results_and_speed, action, reward, next_results_and_speed, logprob_a, done, 0, traj_lenth)
					traj_lenth += 1

					now_state = next_state
					now_speed = next_speed 
				else:
					done = True
			else:
				next_state, reward, done, next_speed, info = env.step(first_action)
				now_state = next_state
				now_speed = next_speed
			print(f"Total reward: {run_reward}") 
			print(f"Total training : {total_train}")
			if traj_lenth >= agent.T_horizon:
				traj_lenth = 0
				total_train += 1
				agent.train()
		if run_reward > best_reward:
			env.close()
			best_reward = save_model(agent, best_reward)
	
	

if __name__ == "__main__":
	main()