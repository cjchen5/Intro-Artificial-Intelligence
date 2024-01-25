# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Initialize the stack with the start state
    start_state = problem.getStartState()
    # Define the stack , and add the initial state and empty action list to the stack
    stack = [(start_state, [])]
    reached = set()

    while stack:
        # Pop the top element of the stack in each loop, and store it in the reached
        state, actions = stack.pop()
        if state in reached:
            continue
        reached.add(state)

        if problem.isGoalState(state):
            # Return the actions needed
            return actions

        for next_state, action, _ in problem.getSuccessors(state):
            # Get the next possible states
            if next_state not in reached:
                # Add the next state to the stack
                stack.append((next_state, actions + [action]))
    # If no solution was found, return None
    return None


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Initialize the queue with the start state
    start_state = problem.getStartState()
    frontier = util.Queue()
    frontier.push((start_state, []))
    # Keep track of the states reached
    reached = set()
    reached.add(start_state)

    while not frontier.isEmpty():
        # Get the next state
        state, actions = frontier.pop()
        if problem.isGoalState(state):
            # Return the actions needed
            return actions
        for next_state, action, _ in problem.getSuccessors(state):
            # Get the next possible states
            if next_state not in reached:
                # Add the next state to the queue and mark it as reached
                frontier.push((next_state, actions + [action]))
                reached.add(next_state)
    # If no solution was found, return None
    return None

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Create a priority queue to store states
    frontier = util.PriorityQueue()
    # Get the start state and add it to the frontier
    start_state = problem.getStartState()
    frontier.push((start_state, 0, []), 0)
    # Keep track of where each state came from and the cost so far
    came_from = dict()
    cost_so_far = dict()
    came_from[start_state] = None
    cost_so_far[start_state] = 0
    # Repeat until the frontier is empty
    while not frontier.isEmpty():
        # Get the next state to expand and its cost and actions
        current_state, current_cost, actions = frontier.pop()
        # If the current state is the goal, return the solution
        if problem.isGoalState(current_state):
            return actions
        # Otherwise, expand the node by getting its successors
        for next_state, action, cost in problem.getSuccessors(current_state):
            # Calculate the new cost to reach the next state
            new_cost = current_cost + cost
            # If this is the first time visiting this state, or if the new cost is lower than the previous cost
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost
                frontier.push((next_state, new_cost, actions + [action]), priority)
                came_from[next_state] = current_state

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    start_state = problem.getStartState()
    # Push the start state into the priority queue, with priority value as heuristic value from start state to goal state.
    frontier.push((start_state, 0, []), heuristic(start_state, problem))
    # Dictionary to store the parent of each state that is processed.
    came_from = dict()
    # Dictionary to store the cost so far to reach each state.
    cost_so_far = dict()
    came_from[start_state] = None
    cost_so_far[start_state] = 0

    while not frontier.isEmpty():
        # Pop the state from priority queue which has the lowest combined cost and heuristic value.
        current_state, current_cost, actions = frontier.pop()

        if problem.isGoalState(current_state):
            # If the current state is the goal state, return the list of actions to reach here.
            return actions

        for next_state, action, cost in problem.getSuccessors(current_state):
            # Calculate the cost so far to reach next state from start state.
            new_cost = current_cost + cost
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                # If next state has not been processed yet or a cheaper path is found, update the cost so far.
                cost_so_far[next_state] = new_cost
                # Calculate the priority value as the sum of cost so far and heuristic value from next state to goal state.
                priority = new_cost + heuristic(next_state, problem)
                # Push the next state into the priority queue, with priority value as the sum of cost so far and heuristic value.
                frontier.push((next_state, new_cost, actions + [action]), priority)
                # Update the parent of next state as the current state.
                came_from[next_state] = current_state

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
