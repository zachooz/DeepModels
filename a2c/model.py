import tensorflow as tf
import numpy as np
import gym


class A2C:
    def __init__(self):
        self.game = gym.make('CartPole-v1')
        self.num_actions = self.game.action_space.n
        self.state_size = self.game.observation_space.shape[0]
        
        # list of game states, rewards, actions for entire game in order
        self.state_input = tf.placeholder(tf.float32, [None, self.state_size], name="state_input")
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="rewards")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        self.learning_rate = 0.004
        self.discount_reward_by = 0.99
        
        # The predicted values for the game states
        self.state_value = self.critic()
        
        # The action probabilites for the game states so len will be num_actions * len(state_input)]
        self.actor_probs = self.actor()
        
        # builds a list of the indexes of actions taken in self.actor_probs
        # [0, num_actions, 2 * num_actions, ....., num states * num_actions] + action take should properly generate this list
        self.indicies = tf.range(0, tf.shape(self.actor_probs)[0]) * self.num_actions + self.actions
        
        # get probability of taken actions
        self.actProbs = tf.gather(tf.reshape(self.actor_probs, [-1]), self.indicies)
        
        # losses
        
        # advantage is the differnce between the discounted reward and the reward we predicted for a each given state
        self.advantage = self.rewards - self.state_value
        
        # if advantage is negative, the predicted reward was greater than the actual, so we want to lower actProbs
        # if advantage is positive, the predicted reward was less than the actual, so we want to raise actProbs
        self.actor_loss = tf.reduce_mean(-tf.log(self.actProbs) * self.advantage)
        
        # MSE works because we want the predicted values to be the same
        self.critic_loss = tf.losses.mean_squared_error(self.rewards, tf.reshape(self.state_value, [-1]))

        self.loss_val = self.loss()
        self.train_op = self.optimizer()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_val)

    def critic(self):
        """
        Estimates input value
        :return: A tensor of shape [num_states] the estimated values of each state
        """
        w1 = tf.Variable(tf.random_normal([self.state_size, self.state_size], stddev=0.1))
        b1 = tf.Variable(tf.random_normal([self.state_size]))
        o1 = tf.nn.elu(tf.add(tf.matmul(self.state_input, w1), b1))

        w2 = tf.Variable(tf.random_normal([self.state_size, 1], stddev=0.1))
        b2 = tf.Variable(tf.random_normal([1]))
        o2 = tf.nn.elu(tf.add(tf.matmul(o1, w2), b2))

        return o2

    def actor(self):
        """
        :return: A tensor of shape [num_states, num_actions] representing the probability distribution
            over actions that is generated by your actor.
        """
        w1 = tf.Variable(tf.random_normal([self.state_size, self.state_size], stddev=0.1))
        b1 = tf.Variable(tf.random_normal([self.state_size]))
        o1 = tf.nn.elu(tf.add(tf.matmul(self.state_input, w1), b1))

        w2 = tf.Variable(tf.random_normal([self.state_size, self.num_actions], stddev=0.1))
        b2 = tf.Variable(tf.random_normal([self.num_actions]))
        o2 = tf.nn.softmax(tf.add(tf.matmul(o1, w2), b2))

        return o2

    def loss(self):
        return self.actor_loss + self.critic_loss

    def train_episode(self):
        # run an entire game
        state = self.game.reset()
        states = []
        rewards = []
        actions = []
        for i in range(999):
            actDist = self.session.run(self.actor_probs, feed_dict={self.state_input: [state]})
            action = np.random.choice(self.num_actions, 1, p=actDist[0])[0]
            next_state, reward, done, info = self.game.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if (done):
                break

        # print(np.sum(rewards))
        
        # discount rewards from game
        rewards = self.discount_rewards(rewards[:])
        
        # train on game result
        self.session.run(self.train_op, feed_dict={self.state_input: states, self.rewards: rewards, self.actions: actions})

    def discount_rewards(self, rewards):
        gamma = self.discount_reward_by
        for i, e in reversed(list(enumerate(rewards))):
            if i + 1 == len(rewards):
                continue
            if rewards[i + 1] == 0:
                break
            rewards[i] = rewards[i] + gamma * rewards[i + 1]
        return rewards

    def visualize(self):
        state = self.game.reset()
        for i in range(999):
            self.game.render()
            actDist = self.session.run(self.actor_probs, feed_dict={self.state_input: [state]})
            action = np.argmax(actDist[0])
            next_state, reward, done, info = self.game.step(action)
            state = next_state
            if (done):
                break
        self.game.close()

    def test(self, num):
        rewards = []
        for t in range(num):
            instance_rewards = []
            state = self.game.reset()
            for i in range(999):
                actDist = self.session.run(self.actor_probs, feed_dict={self.state_input: [state]})
                action = np.argmax(actDist[0])
                next_state, reward, done, info = self.game.step(action)
                instance_rewards.append(reward)
                state = next_state
                if (done):
                    break
            rewards.append(np.sum(instance_rewards))
        return rewards


def check_actor(model):
    dummy_state = np.ones((10, 4))
    actor_probs = model.session.run(model.actor_probs, feed_dict={
        model.state_input: dummy_state
    })
    return actor_probs.shape == (10, 2)


if __name__ == '__main__':
    learner = A2C()
    for i in range(1000):
        learner.train_episode()
    assert(check_actor(learner))
    print(np.mean(learner.test(100)))
