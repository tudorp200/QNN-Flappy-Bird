import gymnasium as gym
import torch
import cv2
import numpy as np
import os
import time

from test import Agent, EnvProcessor

def watch_agent():
    print("Initializing Environment...")
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    env = EnvProcessor(env)
    
    agent = Agent(env)
    
    if os.path.exists("flappy_checkpoint.pth"):
        agent.load("flappy_checkpoint.pth")
        print("Model loaded successfully.")
    else:
        print("Error: No checkpoint found! Train the agent first.")
        return

    agent.epsilon = 0.0
    print(f"Agent Ready! Epsilon: {agent.epsilon}")
    
    episodes = 20
    for ep in range(episodes):
        state, _ = env.reset()
        current_score = 0 
        total_reward = 0

        print(f"Starting Episode {ep+1}...")
        
        while True:
            action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            current_score = info.get('score', 0)
            total_reward += reward

            human_view = env.render() 
            
            if human_view is not None:
                human_view = cv2.cvtColor(human_view, cv2.COLOR_RGB2BGR)
                
                text = f"Score: {current_score}"
                cv2.putText(human_view, text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("Flappy Bird (Agent)", human_view)
            
            if cv2.waitKey(25) == ord('q'): 
                print("Quitting...")
                env.close()
                cv2.destroyAllWindows()
                return

            state = next_state
            
            if done:
                print(f"Episode {ep+1} Finished. Final Score: {current_score}. Reward: {total_reward}")
                time.sleep(1) 
                break
                
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    watch_agent()
