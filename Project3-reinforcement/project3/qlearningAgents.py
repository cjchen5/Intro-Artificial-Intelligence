# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter()  # A counter is a dict with default 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Get all legal actions for the current state
        legal_actions = self.getLegalActions(state)

        # If there are no legal actions, return 0
        if not legal_actions:
            return 0.0

        # Compute the Q-value for each legal action
        q_values = []
        for action in legal_actions:
            q_values.append(self.getQValue(state, action))

        # Return the maximum Q-value
        return max(q_values)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get a list of all the legal actions in the current state
        legal_actions = self.getLegalActions(state)

        # If there are no legal actions, return None
        if not legal_actions:
            return None

        # Set the best action and its value to None and negative infinity respectively
        best_action = None
        best_value = float('-inf')

        # Loop over all legal actions and find the one with the highest Q-value
        for action in legal_actions:
            # Get the Q-value of the current action in the current state
            value = self.getQValue(state, action)
            # If the Q-value is higher than the current best value, update the best action and value
            if value > best_value:
                best_value = value
                best_action = action

        # Return the action with the highest Q-value
        return best_action

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get the legal actions available for the current state
        legalActions = self.getLegalActions(state)

        # If there are no legal actions available, return None
        if len(legalActions) == 0:
            return None

        # Initialize the maximum Q-value to the lowest possible value, the best action to None
        best_value = float('-inf')
        bestAction = None

        # Loop through all legal actions and calculate their Q-values
        for action in legalActions:
            qValue = self.getQValue(state, action)

            # If the current action has a higher Q-value than the current maximum Q-value, update the maximum Q-value and the best action accordingly
            if qValue > best_value:
                best_value = qValue
                bestAction = action

        # Return the best action with the highest Q-value
        return bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        action = None

        # If there are no legal actions, return None
        if not legalActions:
            return None

        # With probability epsilon, explore and choose a random action
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            # Otherwise, choose the action that has the highest Q-value
            q_values = []
            for action in legalActions:
                q_values.append(self.getQValue(state, action))

            max_q_value = max(q_values)

            best_actions = []
            for action, q_values in zip(legalActions, q_values):
                if q_values == max_q_value:
                    best_actions.append(action)
            # If there are multiple actions with the same max Q-value, randomly choose one
            action = random.choice(best_actions)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # The Q-value of the next state
        next_q_value = self.getValue(nextState)
        # The estimated value of taking the action and transitioning to the next state
        sample = reward + self.discount * next_q_value
        # The current Q-value of the state-action pair
        current_q_value = self.getQValue(state, action)
        # The updated Q-value after incorporating the new information
        new_q_value = (1 - self.alpha) * current_q_value + self.alpha * sample
        # Update the Q-value dictionary with the new Q-value
        self.q_values[(state, action)] = new_q_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        # Get features for the current state and action
        features = self.featExtractor.getFeatures(state, action)
        # Loop through each feature and add its weighted value to the Q-value
        for feature in features:
            # Get the weight for the current feature and multiply it by the value of the feature
            weight = self.weights[feature]
            value = features[feature]
            q_value += weight * value
        # Return the final Q-value
        return q_value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Calculate the difference between the expected Q-value and the actual Q-value
        expected_Q_value = reward + self.discount * self.getValue(nextState)
        actual_Q_value = self.getQValue(state, action)
        difference = expected_Q_value - actual_Q_value

        # Get the feature vector for the current state-action pair
        feature_vector = self.featExtractor.getFeatures(state, action)

        # Update the weights of the features using the difference and the learning rate alpha
        for feature, value in feature_vector.items():
            # Update the weight of the feature based on its value and the difference
            self.weights[feature] += self.alpha * difference * value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)
            pass

