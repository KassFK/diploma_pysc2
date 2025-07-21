import tensorflow as tf
import numpy as np
from a2c_model import A2CNetwork
import random

class A2CAgent:
    def __init__(self, state_shape, num_actions, screen_size=84, learning_rate=0.0007, 
                 gamma=0.99, entropy_coef=0.01, value_loss_coef=0.5):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.screen_size = screen_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        
        # Create A2C network
        self.network = A2CNetwork(num_actions, screen_size)
        
        # Initialize network with a dummy forward pass
        dummy_state = np.zeros((1,) + state_shape)
        self.network(dummy_state)
        
        # Use Adam optimizer with a lower learning rate for stability
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # For tracking learning progress
        self.episode_rewards = []
        
    def select_action(self, state, available_actions, training=False):
        """Select action based on the policy network, respecting available actions."""
        if len(available_actions) < 1:
            return 0, {}  # Return no_op action if nothing is available
        
        # Prepare state for network input
        state = np.expand_dims(state, axis=0)
        
        # Get network predictions
        outputs = self.network(state)
        
        # Get action probabilities and mask unavailable actions
        action_logits = outputs['action_logits'][0].numpy()
        
        # Set logits for unavailable actions to large negative value
        mask = np.ones_like(action_logits) * (-1e10)
        mask[available_actions] = 0
        masked_logits = action_logits + mask
        
        # Convert to probabilities
        action_probs = tf.nn.softmax(masked_logits).numpy()
        
        if training:
            # Proper exploration: Sample from probability distribution over available actions
            # First normalize probabilities for available actions
            available_probs = action_probs[available_actions]
            if np.sum(available_probs) > 0:  # Check we have valid probabilities
                available_probs = available_probs / np.sum(available_probs)
                action_idx = np.random.choice(len(available_actions), p=available_probs)
                action = available_actions[action_idx]
            else:
                # Fallback to uniform random if probabilities are invalid
                action = random.choice(available_actions)
        else:
            # In evaluation mode, take the most probable action
            # print(f"test:")
            action = np.argmax(action_probs)
            
        # Get arguments predictions
        args_dict = {}
        try:
            # Get predicted queue argument
            queue_probs = tf.nn.softmax(outputs['queue_logits'][0]).numpy()
            queue_arg = np.random.choice([0, 1], p=queue_probs) if training else np.argmax(queue_probs)
            
            # Get predicted screen coordinates
            screen_x_probs = tf.nn.softmax(outputs['screen_x_logits'][0]).numpy()
            screen_y_probs = tf.nn.softmax(outputs['screen_y_logits'][0]).numpy()
            
            if training:
                x1 = np.random.choice(self.screen_size, p=screen_x_probs)
                y1 = np.random.choice(self.screen_size, p=screen_y_probs)
            else:
                x1 = np.argmax(screen_x_probs)
                y1 = np.argmax(screen_y_probs)
                
            # Get predicted second screen coordinates
            screen2_x_probs = tf.nn.softmax(outputs['screen2_x_logits'][0]).numpy()
            screen2_y_probs = tf.nn.softmax(outputs['screen2_y_logits'][0]).numpy()
            
            if training:
                x2 = np.random.choice(self.screen_size, p=screen2_x_probs)
                y2 = np.random.choice(self.screen_size, p=screen2_y_probs)
            else:
                x2 = np.argmax(screen2_x_probs)
                y2 = np.argmax(screen2_y_probs)
            
            args_dict = {
                'queue': [queue_arg],
                'screen': [x1, y1],
                'screen2': [x2, y2],
                'action_prob': action_probs[action],
                'value': outputs['value'][0][0].numpy()
            }
        except Exception as e:
            print(f"Error extracting arguments: {e}")
            
        # print(f"Action: {action}, Args: {args_dict}")    
        return action, args_dict
        
    def train(self, states, actions, rewards, next_states, dones, action_args):
        """Train the A2C network on a batch of experience."""
        if len(states) == 0:
            return {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'args_loss': 0}
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Get next state values for bootstrapping
        next_values = self.network(next_states)['value']
        next_values = tf.squeeze(next_values, axis=1)
        
        # Compute TD targets and advantages
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        
        with tf.GradientTape() as tape:
            # Forward pass through the network
            outputs = self.network(states)
            values = tf.squeeze(outputs['value'], axis=1)
            action_logits = outputs['action_logits']
            action_probs = outputs['action_probs']
            
            # Calculate advantages
            advantages = tf.stop_gradient(td_targets - values)
            
            # Policy loss (negative log probability weighted by advantage)
            action_one_hot = tf.one_hot(actions, self.num_actions)
            log_probs = tf.reduce_sum(tf.math.log(action_probs + 1e-10) * action_one_hot, axis=1)
            policy_loss = -tf.reduce_mean(log_probs * advantages)
            
            # Value loss (MSE)
            value_loss = tf.reduce_mean(tf.square(td_targets - values))
            
            # Entropy loss (for exploration)
            entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1))
            
            # Arguments losses
            args_loss = 0.0
            
            for i, args in enumerate(action_args):
                if not args:  # Skip if no args are provided
                    continue
                    
                try:
                    # Queue argument loss
                    if 'queue' in args and args['queue']:
                        queue_val = tf.convert_to_tensor([args['queue'][0]], dtype=tf.int32)
                        queue_logits = outputs['queue_logits'][i]
                        queue_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True)(queue_val, tf.expand_dims(queue_logits, 0))
                        args_loss += 0.1 * queue_loss
                        
                    # Screen coordinates losses
                    if 'screen' in args and len(args['screen']) == 2:
                        x_val = tf.convert_to_tensor([args['screen'][0]], dtype=tf.int32)
                        y_val = tf.convert_to_tensor([args['screen'][1]], dtype=tf.int32)
                        
                        x_logits = outputs['screen_x_logits'][i]
                        y_logits = outputs['screen_y_logits'][i]
                        
                        x_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True)(x_val, tf.expand_dims(x_logits, 0))
                        y_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True)(y_val, tf.expand_dims(y_logits, 0))
                        
                        args_loss += 0.1 * (x_loss + y_loss)
                        
                    # Second screen coordinates losses
                    if 'screen2' in args and len(args['screen2']) == 2:
                        x2_val = tf.convert_to_tensor([args['screen2'][0]], dtype=tf.int32)
                        y2_val = tf.convert_to_tensor([args['screen2'][1]], dtype=tf.int32)
                        
                        x2_logits = outputs['screen2_x_logits'][i]
                        y2_logits = outputs['screen2_y_logits'][i]
                        
                        x2_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True)(x2_val, tf.expand_dims(x2_logits, 0))
                        y2_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True)(y2_val, tf.expand_dims(y2_logits, 0))
                        
                        args_loss += 0.1 * (x2_loss + y2_loss)
                except Exception as e:
                    print(f"Error processing action args for loss: {e}")
            
            # Total loss
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy + args_loss
            
        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.network.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        
        return {
            'total_loss': total_loss.numpy(),
            'policy_loss': policy_loss.numpy(),
            'value_loss': value_loss.numpy(),
            'entropy': entropy.numpy(),
            'args_loss': args_loss
        }
        
    def save(self, path):
        self.network.save_weights(path)
        
    def load(self, path):
        self.network.load_weights(path)
