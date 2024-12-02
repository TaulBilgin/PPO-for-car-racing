Custom Modifications to CarRacing-v2:
Step 1: Environment Modification
I added a calculation for the car's true speed by inserting the following line at line 586:
true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))

I modified the return statement to include true_speed:
return self.state, step_reward, terminated, true_speed, info

Step 2: Object Detection for Distance Measurement
I built an object detection system using YOLO to detect the car's position in the environment.
This system measures the distance between the car and the nearest green pixels degree on the track.

Training Focus:
I limited the car's speed and focused on training the agent solely for steering control to simplify the learning process and reduce training time.
