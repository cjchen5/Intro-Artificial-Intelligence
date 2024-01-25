# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            # Initialize a temporary counter to hold new values
            new_values = util.Counter()

            # For each state in the MDP
            for state in self.mdp.getStates():
                # If the state is terminal, set its value to 0
                if self.mdp.isTerminal(state):
                    new_values[state] = 0
                else:
                    # Initialize the best value to be negative infinity
                    best_value = float("-inf")
                    # For each action possible from the state
                    for action in self.mdp.getPossibleActions(state):
                        # Compute the Q-value of the action
                        q_value = self.computeQValueFromValues(state, action)
                        # Update the best value if the Q-value is greater than the current best value
                        if q_value > best_value:
                            best_value = q_value
                    # Set the new value of the state to the best value
                    new_values[state] = best_value

            # Update the current values with the new values
            self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Get all possible next states and their corresponding probabilities for a given state and action
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        Q_value = 0

        # Loop over all possible transition states and probabilities
        for next_state, prob in transitions:
            # Get the reward for transitioning from the current state to the next state
            reward = self.mdp.getReward(state, action, next_state)

            # Get the value function estimate for the next state
            next_state_value = self.values[next_state]

            # Calculate the Q-value for the current state and action
            Q_value += prob * (reward + self.discount * next_state_value)

        # Return the Q-value for the current state and action
        return Q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        # Initialize the best action to None and the best value to negative infinity
        best_action = None
        best_value = float("-inf")

        # For each action possible from the state
        for action in self.mdp.getPossibleActions(state):
            # Compute the Q-value of the action
            q_value = self.computeQValueFromValues(state, action)
            # Update the best action and best value if the Q-value is greater than the current best value
            if q_value > best_value:
                best_action = action
                best_value = q_value

        # Return the best action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Get all possible states in the MDP
        states = self.mdp.getStates()

        for i in range(self.iterations):

            # Select the current state based on the iteration number
            state = states[i % len(states)]

            if self.mdp.isTerminal(state):
                continue

            # Initialize the best value as negative infinity
            best_value = float('-inf')

            # Loop over all possible actions in the current state
            for action in self.mdp.getPossibleActions(state):

                # Compute the Q-value for the current state and action
                q_value = self.computeQValueFromValues(state, action)

                # Update the best value if the current Q-value is higher
                if q_value > best_value:
                    best_value = q_value

            # Update the value function for the current state to the best Q-value
            self.values[state] = best_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states
        predecessors = {state: set() for state in self.mdp.getStates()}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[nextState].add(state)

        # Initialize an empty priority queue
        priorityQueue = util.PriorityQueue()

        # For each state s, do:
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # Find the absolute difference between the current value of state in self.values and the highest Q-value across all possible actions from state
                bestQValue = max(
                    [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                diff = abs(self.values[state] - bestQValue)

                # Add state to the priority queue with priority -diff
                priorityQueue.update(state, -diff)

        # Compute predecessors of all states
        predecessors = {s: set() for s in self.mdp.getStates()}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[nextState].add(state)

        # Initialize an empty priority queue
        priorityQueue = util.PriorityQueue()

        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for iteration in range(self.iterations):
            # For each state, do:
            for state in self.mdp.getStates():
                # If state is a terminal state, skip to the next state
                if self.mdp.isTerminal(state):
                    continue

                # Compute the Q-value for each action in the state
                qValues = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]

                # Compute the absolute difference between the current value of the state in self.values and the highest Q-value across all possible actions from the state
                diff = abs(self.values[state] - max(qValues))

                # Add the state to the priority queue with priority -diff
                priorityQueue.update(state, -diff)

            # If the priority queue is empty, terminate
            if priorityQueue.isEmpty():
                break

            # Pop a state off the priority queue
            poppedState = priorityQueue.pop()

            # Update the popped state's value (if it is not a terminal state) in self.values
            if not self.mdp.isTerminal(poppedState):
                qValues = [self.computeQValueFromValues(poppedState, action) for action in
                           self.mdp.getPossibleActions(poppedState)]
                self.values[poppedState] = max(qValues)

            # For each predecessor of the popped state, do:
            for predecessor in predecessors[poppedState]:
                # If the predecessor is a terminal state, skip to the next predecessor
                if self.mdp.isTerminal(predecessor):
                    continue

                # Compute the Q-value for each action in the predecessor
                qValues = [self.computeQValueFromValues(predecessor, action) for action in
                           self.mdp.getPossibleActions(predecessor)]

                # Compute the absolute difference between the current value of the predecessor in self.values and the highest Q-value across all possible actions from the predecessor
                diff = abs(self.values[predecessor] - max(qValues))

                # If diff > theta, add the predecessor to the priority queue with priority -diff
                if diff > self.theta:
                    priorityQueue.update(predecessor, -diff)
