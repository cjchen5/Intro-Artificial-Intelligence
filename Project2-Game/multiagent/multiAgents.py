# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Get the remaining food in the new state
        newFood = newFood.asList()

        # Get the positions of the ghosts in the new state
        ghostPositions = []

        # Iterate over each ghost state
        for ghostState in newGhostStates:
            # Get the position of the ghost in the current state
            ghostPos = ghostState.getPosition()
            # Add the ghost position to the list
            ghostPositions.append(ghostPos)

        ghostPos = []
        for Ghost in ghostPositions:
            x, y = Ghost[0], Ghost[1]
            ghostPos.append((x, y))

        # Check if Pacman is scared and can eat the ghosts
        scared = min(newScaredTimes) > 0

        # If Pacman will die, return the lowest value
        if not scared:
            ghostPositions = []
            for ghost in newGhostStates:
                x, y = ghost.getPosition()
                ghostPositions.append((x, y))
                if newPos in ghostPositions:
                    return -float("inf")

        # If Pacman will eat food, return the highest value
        foodGrid = currentGameState.getFood()
        foodList = foodGrid.asList()
        if newPos in foodList:
            return float("inf")

        def distance(x):
            return manhattanDistance(x, newPos)

        # Calculate distances to nearest food
        try:
            nearestFood = min(newFood, key=distance)
            closestFoodDist = manhattanDistance(newPos, nearestFood)
        except ValueError:
            closestFoodDist = 0

        # Calculate distances to ghost
        try:
            ghostDistances = min(ghostPos, key=distance)
            closestGhostDist = manhattanDistance(newPos, ghostDistances)
        except ValueError:
            closestGhostDist = float("inf")

        # Calculate evaluation function
        if closestGhostDist == 0:
            # Ghost is on top of Pacman
            return -float("inf")
        elif closestGhostDist < 2 and scared:
            # If Pacman is scared and near a ghost, encourage him to get closer
            ghostPriority = 1.0 / closestGhostDist
            foodPriority = 1.0 / closestFoodDist
            return ghostPriority + foodPriority
        else:
            # Otherwise, encourage him to get closer to the nearest food and farther from the nearest ghost
            foodPriority = 1.0 / closestFoodDist
            ghostPriority = -1.0 / closestGhostDist
            return foodPriority + ghostPriority

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # get legal actions for each ghost
        numAgents = gameState.getNumAgents()
        ghosts = [i for i in range(1, numAgents)]

        # terminal state function:# Check if the state is terminal (win or lose) or if the maximum search depth has been reached
        def is_terminal(state, depth):
            is_win = state.isWin()
            is_lose = state.isLose()
            is_max_depth = depth == self.depth
            return is_win or is_lose or is_max_depth

        # minimizer function
        def min_value(state, depth, ghost):
            if is_terminal(state, depth):
                return self.evaluationFunction(state)

            value = float("inf")  # initialize to positive infinity
            for action in state.getLegalActions(ghost):
                if ghost == ghosts[-1]:
                    successor = state.generateSuccessor(ghost, action)
                    new_value = max_value(successor, depth + 1)  # calculate the max value of the successor state
                    value = min(value, new_value)  # update the current value with the minimum of current and new value
                else:
                    next_state = state.generateSuccessor(ghost, action)
                    next_ghost = ghost + 1
                    value = min(value, min_value(next_state, depth, next_ghost))
            return value

        # maximizer function
        def max_value(state, depth):
            if is_terminal(state, depth):
                return self.evaluationFunction(state)

            value = float("-inf")  # initialize to negative infinity
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value = max(value, min_value(successor, depth, 1))
            return value

        # select action for the maximizer
        best_action, best_value = None, float("-inf")
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            value = min_value(successorState, 0, 1)
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # terminal state function:# Check if the state is terminal (win or lose) or if the maximum search depth has been reached
        def is_terminal(state, depth):
            is_win = state.isWin()
            is_lose = state.isLose()
            is_max_depth = depth == self.depth
            return is_win or is_lose or is_max_depth

        def min_value(state, depth, ghost_index, alpha, beta):
            # Check if the state is terminal
            if is_terminal(state, depth):
                return self.evaluationFunction(state)

            # Initialize the value to infinity
            value = float('inf')

            # Loop through all legal actions for the current ghost
            for action in state.getLegalActions(ghost_index):
                # Generate the successor state for the current action
                successor = state.generateSuccessor(ghost_index, action)

                # If we've reached the last ghost, call the max_value function for the next level
                if ghost_index == state.getNumAgents() - 1:
                    value = min(value, max_value(successor, depth + 1, alpha, beta))
                # Otherwise, call the min_value function for the next ghost
                else:
                    value = min(value, min_value(successor, depth, ghost_index + 1, alpha, beta))

                # If the current value is less than alpha, we can prune this branch
                if value < alpha:
                    return value

                # Update the beta value
                beta = min(beta, value)

            # Return the final value
            return value

        def max_value(state, depth, alpha, beta):
            if is_terminal(state, depth):
                return self.evaluationFunction(state)

            value = float("-inf")
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value = max(value, min_value(successor, depth, 1, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        # Get legal actions for the current state
        legal_actions = gameState.getLegalActions(0)

        # Initialize variables
        best_action = None
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        # Evaluate each possible action and update best action and score
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(0, action)
            score = min_value(successor_state, 0, 1, alpha, beta)

            # Update best action and score if new score is better
            if score > best_score:
                best_action = action
                best_score = score

            # Update alpha with best score so far
            alpha = max(alpha, best_score)

        # Return best action
        return best_action

        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Define the indices of the ghosts
        numAgents = gameState.getNumAgents()
        ghost_indices = [i for i in range(1, numAgents)]

        # terminal state function:# Check if the state is terminal (win or lose) or if the maximum search depth has been reached
        def is_terminal(state, depth):
            is_win = state.isWin()
            is_lose = state.isLose()
            is_max_depth = depth == self.depth
            return is_win or is_lose or is_max_depth

        # Calculate the expected value for the current ghost (minimizer)
        def exp_value(state, depth, ghost):

            # Check if the state is terminal
            if is_terminal(state, depth):
                return self.evaluationFunction(state)

            # Initialize the expected value
            expected_value = 0

            # Calculate the probability of each action for the current ghost
            legal_actions = state.getLegalActions(ghost)
            num_legal_actions = len(legal_actions)
            probability = 1 / num_legal_actions

            # Calculate the expected value as the weighted sum of the values of each successor state
            for action in state.getLegalActions(ghost):
                next_state = state.generateSuccessor(ghost, action)

                if ghost == ghost_indices[-1]:
                    # If this is the last ghost, then the next level is the max level
                    next_value = max_value(next_state, depth + 1)
                else:
                    # Otherwise, the next level is another ghost
                    next_value = exp_value(next_state, depth, ghost + 1)

                next_expected_value = probability * next_value
                expected_value += next_expected_value
            return expected_value

        def max_value(state, depth):
            # Calculate the maximum value for pacman (maximizer)

            # Check if the state is terminal
            if is_terminal(state, depth):
                return self.evaluationFunction(state)

            # Initialize the maximum value
            max_value = -float('inf')

            # Calculate the maximum value as the maximum value of all successor states
            for action in state.getLegalActions(0):
                next_state = state.generateSuccessor(0, action)
                next_expected_value = exp_value(next_state, depth, 1)
                max_value = max(max_value, next_expected_value)
            return max_value

        # Calculate the expectimax values for each possible action
        expectimax_values = []
        for action in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, action)
            expected_value = exp_value(next_state, 0, 1)
            expectimax_values.append((action, expected_value))

        # Sort the actions based on their expectimax values
        def get_second_element(pair):
            return pair[1]
        expectimax_values.sort(key=get_second_element)

        # Return the action with the highest expectimax value
        return expectimax_values[-1][0]

        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Get current state information
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    # Evaluate food score: Check if there is any food left
    foodList = Food.asList()
    if len(foodList) > 0:
        # Calculate the distance to the nearest food
        distancesToFood = []
        for foodPos in foodList:
            distance = manhattanDistance(Pos, foodPos)
            distancesToFood.append(distance)
        nearestFoodDistance = min(distancesToFood)
        # Evaluate the food score based on the distance
        foodScore = 1 / nearestFoodDistance
    else:
        # If there is no food, set the food score to 0
        foodScore = 0

    # Evaluate danger score: Calculate the distance to the nearest ghost
    distancesToGhost = []
    for ghostState in GhostStates:
        distance = manhattanDistance(Pos, ghostState.configuration.pos)
        distancesToGhost.append(distance)
    nearestGhostDistance = min(distancesToGhost)

    # Evaluate the danger score based on the distance
    if nearestGhostDistance != 0:
        dangerScore = -2 / nearestGhostDistance
    else:
        dangerScore = 0

    # Evaluate total scared time
    totalScaredTimes = sum(ScaredTimes)

    # Calculate the final score
    current_Score = currentGameState.getScore()
    final_Score = current_Score + foodScore + dangerScore + totalScaredTimes

    return final_Score
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
