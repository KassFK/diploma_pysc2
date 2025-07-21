from sc2_env import SC2Env
from a2c_agent import A2CAgent
import numpy as np
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from absl import flags
from absl import app

FLAGS = flags.FLAGS

def train_a2c():
    # Environment setup
    env = SC2Env(map_name="MoveToBeacon")
    state_shape = (3, 84, 84)
    num_actions = 524
    
    # A2C agent with tuned hyperparameters
    agent = A2CAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        learning_rate=0.0005,  # Lower learning rate for stability
        gamma=0.99,            # Discount factor
        entropy_coef=0.01,     # Encourages exploration
        value_loss_coef=0.5    # Balances value and policy learning
    )
    
    # Training parameters
    num_episodes = 3
    update_frequency = 8       # Number of steps before updating network
    save_frequency = 50        # Episodes between model saves
    
    # Tracking metrics
    all_rewards = []
    running_reward = deque(maxlen=100)
    all_losses = []
    entropy_history = []
    
    # Model directory
    model_dir = "/Users/thanoskasselas/Documents/Git Code/test_gpt/use/co1/weights/"
    os.makedirs(model_dir, exist_ok=True)
    
    # Find and load latest model if it exists
    latest_model = None
    latest_episode = -1
    
    for filename in os.listdir(model_dir):
        if filename.startswith("sc2_a2c_") and filename.endswith(".weights.h5"):
            try:
                episode_num = int(filename.split("_")[-1].split(".")[0])
                if episode_num > latest_episode:
                    latest_episode = episode_num
                    latest_model = filename
            except ValueError:
                continue

    if latest_model:
        model_path = os.path.join(model_dir, latest_model)
        print(f"Loading model from {model_path}")
        agent.load(model_path)
        start_episode = latest_episode + 1
        print(f"Resuming training from episode {start_episode}")
    else:
        start_episode = 0
        print("No existing model found. Starting training from scratch.")
    
    # Training loop
    total_steps = 0
    start_time = time.time()
    
    for episode in range(start_episode, start_episode + num_episodes):
        # Initialize episode variables
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        step = 0
        
        # For storing trajectory for batch update
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        action_args = []
        
        while not done:
            # Select action based on current policy
            available_actions = env.get_available_actions()

            # Get training false if its 10 episodes from the end
            if episode >= start_episode + num_episodes - 5:
                training = False
            else:
                training = True
            action, args = agent.select_action(state, available_actions, training)
            
            # Take action in environment
            next_state, reward, done, env_args = env.step(action)
            
            # Store step data for batch update
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            action_args.append(args)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step += 1
            total_steps += 1
            
            # Perform batch update every update_frequency steps or at end of episode
            if step % update_frequency == 0 or done:
                if len(states) > 0:
                    loss_dict = agent.train(
                        np.array(states),
                        np.array(actions),
                        np.array(rewards),
                        np.array(next_states),
                        np.array(dones),
                        action_args
                    )
                    episode_loss.append(loss_dict['total_loss'])
                    entropy_history.append(loss_dict['entropy'])
                    
                    # Clear trajectory data after update
                    states = []
                    actions = []
                    rewards = []
                    next_states = []
                    dones = []
                    action_args = []
        
        # Episode completed
        running_reward.append(episode_reward)
        all_rewards.append(episode_reward)
        if episode_loss:
            all_losses.append(np.mean(episode_loss))
        
        avg_reward = np.mean(running_reward)
        
        # Print progress
        print(f"Episode: {episode}, Steps: {step}, Reward: {episode_reward:.2f}, "
              f"Avg Reward: {avg_reward:.2f}, Loss: {np.mean(episode_loss) if episode_loss else 0:.6f}")
        
        # Save model periodically
        if episode % save_frequency == 0 or episode == start_episode + num_episodes - 1:
            save_path = os.path.join(model_dir, f"sc2_a2c_{episode}.weights.h5")
            agent.save(save_path)
            print(f"Model saved to {save_path}")
            
            # Plot and save training progress
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(all_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            plt.subplot(2, 2, 2)
            plt.plot(all_losses)
            plt.title('Training Loss')
            plt.xlabel('Update')
            plt.ylabel('Loss')
            
            plt.subplot(2, 2, 3)
            plt.plot(entropy_history)
            plt.title('Policy Entropy')
            plt.xlabel('Update')
            plt.ylabel('Entropy')
            
            if len(running_reward) > 1:
                plt.subplot(2, 2, 4)
                plt.plot(np.arange(len(running_reward)), running_reward)
                plt.title('Running Average Reward (100 episodes)')
                plt.xlabel('Episode')
                plt.ylabel('Avg Reward')
            
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f"a2c_training_progress_{episode}.png"))
            plt.close()
    
    # Training finished
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/60:.2f} minutes")
    print(f"Final average reward: {np.mean(running_reward):.2f}")
    
    # Close the environment
    if hasattr(env, 'close'):
        env.close()

def main(argv):
    train_a2c()

if __name__ == "__main__":
    app.run(main)
