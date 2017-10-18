import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.env = env
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        self.normalization = normalization
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess
        self._state_dim = env._get_obs().shape[0]
        self._action_dim = env.action_space.sample().shape[0]   #TODO
        self._s = tf.placeholder(tf.float32, [None, self._state_dim])
        self._a = tf.placeholder(tf.float32, [None, self._action_dim])
        self._deltas = tf.placeholder(tf.float32, [None, self._state_dim])
        self._s_a = tf.concat([self._s, self._a], axis=1)
        self.delta_pred = build_mlp(self._s_a, self._state_dim, 'dynamics', n_layers=n_layers, size=size, activation = activation, output_activation=output_activation)
        self.pred_error = tf.reduce_mean(tf.reduce_sum(tf.square(self.delta_pred - self._deltas), reduction_indices = [1]))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.pred_error)
        
    def fit(self, data):
        obs = []
        actions = []
        next_obs = []
        for path in data:
            obs += path['observations']
            actions += path['actions']
            next_obs += path['next_observations']
        obs = np.array(obs)
        next_obs = np.array(next_obs)
        actions = np.array(actions)
        deltas = next_obs - obs
        mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action = self.normalization
        normalized_obs = (obs - mean_obs)/(std_obs + 1e-6)
        normalized_actions = (actions - mean_action)/(std_action + 1e-6)
        normalized_deltas = (deltas - mean_deltas)/(std_deltas + 1e-6)
        
        # Batching and running multiple iterations:
        for train_iter in range(self.iterations):
            indices = np.random.randint(0, normalized_obs.shape[0], self.batch_size)
            batch_obs = normalized_obs[indices,:]
            batch_acts = normalized_actions[indices, :]
            batch_deltas = normalized_deltas[indices,:]
            self.sess.run(self.train_op, feed_dict = {self._s: batch_obs, self._a: batch_acts, self._deltas: batch_deltas})
            print "Dynamics Loss", self.sess.run(self.pred_error, feed_dict={self._s: batch_obs, self._a: batch_acts, self._deltas: batch_deltas})
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        
        mean_state, std_state, mean_deltas, std_deltas, mean_action, std_action = self.normalization
        normalized_states = (states - mean_state)/(std_state + 1e-6)
        normalized_actions = (actions - mean_action)/(std_action + 1e-6)
        delta_pred_normalized = self.sess.run(self.delta_pred, feed_dict = {self._s:normalized_states, self._a:normalized_actions})
        delta_pred = delta_pred_normalized*std_deltas + mean_deltas
        next_states = states + delta_pred
        return next_states
