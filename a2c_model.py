import tensorflow as tf

class A2CNetwork(tf.keras.Model):
    def __init__(self, num_actions, screen_size=84):
        super(A2CNetwork, self).__init__()
        
        # Shared convolutional feature extractor
        self.conv1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.shared_dense = tf.keras.layers.Dense(512, activation='relu')
        
        # Policy network (actor)
        self.policy_dense = tf.keras.layers.Dense(256, activation='relu')
        self.action_logits = tf.keras.layers.Dense(num_actions)  # Outputs logits for each action
        
        # Value network (critic)
        self.value_dense = tf.keras.layers.Dense(256, activation='relu')
        self.value_output = tf.keras.layers.Dense(1)  # Single value output
        
        # Arguments networks
        self.args_dense = tf.keras.layers.Dense(256, activation='relu')
        self.queue_logits = tf.keras.layers.Dense(2)  # For queue argument [0, 1]
        self.screen_x_logits = tf.keras.layers.Dense(screen_size)  # Screen x coordinate
        self.screen_y_logits = tf.keras.layers.Dense(screen_size)  # Screen y coordinate
        self.screen2_x_logits = tf.keras.layers.Dense(screen_size)  # Second screen x coordinate
        self.screen2_y_logits = tf.keras.layers.Dense(screen_size)  # Second screen y coordinate
        
    def call(self, inputs):
        x = tf.cast(inputs, tf.float32) / 255.0
        x = tf.transpose(x, [0, 2, 3, 1])  # NCHW -> NHWC
        
        # Shared feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        shared_features = self.shared_dense(x)
        
        # Actor branch (policy)
        policy_features = self.policy_dense(shared_features)
        action_logits = self.action_logits(policy_features)
        action_probs = tf.nn.softmax(action_logits)
        
        # Critic branch (value function)
        value_features = self.value_dense(shared_features)
        value = self.value_output(value_features)
        
        # Arguments branches
        args_features = self.args_dense(shared_features)
        queue_logits = self.queue_logits(args_features)
        screen_x_logits = self.screen_x_logits(args_features)
        screen_y_logits = self.screen_y_logits(args_features)
        screen2_x_logits = self.screen2_x_logits(args_features)
        screen2_y_logits = self.screen2_y_logits(args_features)
        
        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'value': value,
            'queue_logits': queue_logits,
            'screen_x_logits': screen_x_logits,
            'screen_y_logits': screen_y_logits,
            'screen2_x_logits': screen2_x_logits,
            'screen2_y_logits': screen2_y_logits
        }
