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
model = YOLO("C:\\code\\NLP\\Model\\best.pt").to(device)
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



class PPO_agent():
	def __init__(self):
	
		self.state_dim = 73
		self.action_dim = 1
		self.net_width = 150

		# Choose distribution for the actor
		self.actor = Actor(self.state_dim, self.action_dim, self.net_width).to(device)
		# Load the pre-trained model weights for the Actor network
		self.actor.load_state_dict(torch.load("actor-469.pt")) # like "Pendulum-123.pt"

		# Switch the Actor network to evaluation mode (disables dropout, etc.)
		self.actor.eval()

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
	

def distances_and_speed_tensor(now_results, now_speed):
    # Convert to float32 explicitly
    tensor = torch.tensor(now_results, dtype=torch.float32).to(device).view(-1)
    tensor = torch.cat((tensor, torch.tensor([now_speed], dtype=torch.float32).to(device)))
    return tensor

def apply_green_filter(image):
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define the range for green color
    lower_green = np.array([40, 40, 40])  # Adjust based on your requirements
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

def main():
	env = gym.make("CarRacing-v3", render_mode="human")
	agent = PPO_agent()
	env_seed = 2
	now_speed = 0
	first_action = np.array([0, 0, 0])
	action_tensor = np.array([0, 0.1, 0])
	
	total_steps = 0
	while True:
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
				action, logprob_a = agent.select_action(now_results_and_speed, deterministic=True)
				action_tensor[0] = action
				action_tensor[2] = break_value

				next_state, reward, done, next_speed, info = env.step(action_tensor)
				total_steps += 1  # Increment total steps
				
				next_touch, next_results = detector(next_state)

				if next_touch == True:
					done = True
					break

				run_reward += reward
				
				if next_results != "none":
					reward += next_speed

					now_state = next_state
					now_speed = next_speed 
				else:
					done = True
					break
			else:
				next_state, reward, done, next_speed, info = env.step(first_action)
				now_state = next_state
				now_speed = next_speed
			
	

if __name__ == "__main__":
	main()