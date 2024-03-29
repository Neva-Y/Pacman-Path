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

from cmath import inf
from itertools import accumulate
from queue import PriorityQueue
from math import inf as INF
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


class SearchDirection:
    """
    Used to switch directions for bidirectional search, initialising to a forward search 'F'
    """

    def __init__(self):
        self.dir = 'F'

    def switchDir(self):
        if self.dir == 'F':
            self.dir = 'B'
        elif self.dir == 'B':
            self.dir = 'F'
        else:
            util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


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
    stack = util.Stack()
    startState = problem.getStartState()
    startNode = (startState, '', 0, [])
    stack.push(startNode)
    visited = set()
    while not stack.isEmpty():
        node = stack.pop()
        state, action, cost, path = node
        if state not in visited:
            visited.add(state)
            if problem.isGoalState(state):
                path = path + [(state, action)]
                actions = [action[1] for action in path]
                del actions[0]
                return actions
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                stack.push(newNode)
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    startState = problem.getStartState()
    startNode = (startState, '', 0, [])
    queue.push(startNode)
    visited = set()
    while not queue.isEmpty():
        node = queue.pop()
        state, action, cost, path = node
        if state not in visited:
            visited.add(state)
            if problem.isGoalState(state):
                path = path + [(state, action)]
                actions = [action[1] for action in path]
                del actions[0]
                return actions
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                queue.push(newNode)
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of the least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


# Please DO NOT change the following code, we will use it later
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, '', 0, [])
    myPQ.push(startNode, heuristic(startState, problem))
    visited = set()
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, action, cost, path = node
        if (state not in visited) or cost < best_g.get(state):
            visited.add(state)
            best_g[state] = cost
            if problem.isGoalState(state):
                path = path + [(state, action)]
                actions = [action[1] for action in path]
                del actions[0]
                return actions
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                myPQ.push(newNode, heuristic(succState, problem) + cost + succCost)
    util.raiseNotDefined()


def enforcedHillClimbing(problem, heuristic=nullHeuristic):
    """
    Local search with heuristic function.
    You DO NOT need to implement any heuristic, but you DO have to call it.
    The heuristic function is "manhattanHeuristic" from searchAgent.py.
    It will be pass to this function as second argument (heuristic).
    """
    "*** YOUR CODE HERE FOR TASK 1 ***"
    currNode = (problem.getStartState(), '', 0, [])
    currState, action, cost, path = currNode
    while not problem.isGoalState(currState):
        currNode = ehcImprove(currNode, problem, heuristic)
        currState, action, cost, path = currNode

    if problem.isGoalState(currState):
        path = path + [(currState, action)]
        actions = [action[1] for action in path]
        del actions[0]
        return actions
    else:
        util.raiseNotDefined()


def ehcImprove(currNode, problem, heuristic):
    queue = util.Queue()
    queue.push(currNode)
    currHeuristic = heuristic(currNode[0], problem)
    visited = set()
    while not queue.isEmpty():
        node = queue.pop()
        state, action, cost, path = node
        if state not in visited:
            visited.add(state)
            if heuristic(state, problem) < currHeuristic:
                return node
            else:
                for succ in problem.getSuccessors(state):
                    succState, succAction, succCost = succ
                    newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                    queue.push(newNode)
    util.raiseNotDefined()


def bidirectionalAStarEnhanced(problem, heuristic=nullHeuristic, backwardsHeuristic=nullHeuristic):
    """
    Bidirectional global search with heuristic function.
    You DO NOT need to implement any heuristic, but you DO have to call them.
    The heuristic functions are "manhattanHeuristic" and "backwardsManhattanHeuristic" from searchAgent.py.
    It will be pass to this function as second and third arguments.
    You can call it by using: heuristic(state,problem) or backwardsHeuristic(state,problem)
    """
    "*** YOUR CODE HERE FOR TASK 2 ***"
    # Dictionaries for the forward and backward search to keep track of states in the corresponding priority queues
    openForwardStates = {}
    openBackwardStates = {}

    # Start state initialisation
    startState = problem.getStartState()
    openForward = util.PriorityQueue()
    openForward.push((startState, '', 0, []), heuristic(startState, problem))
    openForwardStates[startState] = 1
    costForward = {}
    closedForwardSet = set()
    pathForward = {}

    # Goal states initialisation
    openBackward = util.PriorityQueue()
    closedBackwardSet = set()
    costBackward = {}
    pathBackward = {}
    goalStates = problem.getGoalStates()
    # Also keep track of the initial goal state for the backwards search
    for goalState in goalStates:
        openBackward.push((goalState, '', 0, [], goalState), backwardsHeuristic(goalState, problem))
        openBackwardStates[(goalState, goalState)] = 1

    # Lower, upper bound and plan initialisation
    L, U = 0, INF
    plan = []

    # Initialise search direction as forward
    searchDir = SearchDirection()

    while not openForward.isEmpty() and not openBackward.isEmpty():

        L = (openForward.getMinimumPriority() + openBackward.getMinimumPriority()) / 2

        # Alternate between forwards and backward search
        if searchDir.dir == 'F':
            state, action, pathCost, path = openForward.pop()

            # Keeping track of the states in the priority queue, note that there could be additional same state in
            # the PQ (potentially better b value) so a dictionary counter is used and adding the state into the closed set
            openForwardStates[state] = openForwardStates[state] - 1
            if openForwardStates[state] == 0:
                del openForwardStates[state]
            closedForwardSet.add(state)

            # Since there can be multiple "food" goals, check if a path has been found ending at any of these states
            for goalState in goalStates:

                # Create the path if the state matches with the forward search state and the cost is better than current path
                if (state, goalState) in openBackwardStates.keys() and pathCost + costBackward[(state, goalState)][0] < U:
                    U = pathCost + costBackward[(state, goalState)][0]
                    forwardActions = [action[1] for action in (path + [(state, action)])]
                    backwardActions = [action[1] for action in reversed(pathBackward[(state, goalState)])]
                    del forwardActions[0]
                    del backwardActions[-1]
                    plan = forwardActions + [costBackward[(state, goalState)][1]] + backwardActions
                    break
        else:
            # Same idea for the backward search
            state, action, pathCost, path, initialGoal = openBackward.pop()
            openBackwardStates[(state, initialGoal)] = openBackwardStates[(state, initialGoal)] - 1
            if openBackwardStates[(state, initialGoal)] == 0:
                del openBackwardStates[(state, initialGoal)]
            closedBackwardSet.add((state, initialGoal))
            if state in openForwardStates.keys() and pathCost + costForward[state][0] < U:
                U = pathCost + costForward[state][0]
                backwardActions = [action[1] for action in reversed(path + [(state, action)])]
                forwardActions = [action[1] for action in pathForward[state]]
                del forwardActions[0]
                del backwardActions[-1]
                plan = forwardActions + [costForward[state][1]] + backwardActions

        # Return the plan if the lower bound is greater or equal to the upper bound
        if L >= U:
            print("Length of plan:", len(plan))
            return plan

        # Alternate between getting forward and backward successor
        if searchDir.dir == 'F':
            for succ in problem.getSuccessors(state):

                # Consider unique states, ignore states that have been popped by the priority queue as it will
                # have a lower b-value and will not need to be considered
                if succ[0] not in closedForwardSet:
                    succState, succAction, succCost = succ

                    # Calculate b-value use this to add the successor node into the priority queue, adding dict counter
                    bValue = 2 * (pathCost + succCost) + heuristic(succState, problem) - backwardsHeuristic(succState, problem)
                    newNode = (succState, succAction, pathCost + succCost, path + [(state, action)])
                    openForward.push(newNode, bValue)
                    if succState in openForwardStates:
                        openForwardStates[succState] = openForwardStates[succState] + 1
                    else:
                        openForwardStates[succState] = 1

                    # Update the best cost and path for the state
                    if succState in costForward.keys() and costForward[succState][0] > (succCost + pathCost):
                        pathForward[succState] = newNode[3]
                        costForward[succState] = [succCost + pathCost, succAction]
                    elif succState not in costForward.keys():
                        pathForward[succState] = newNode[3]
                        costForward[succState] = [succCost + pathCost, succAction]
        else:
            # Similar idea to backward successor but also encoding the initial goal state information to store best
            # path and cost for that initial goal state
            for succ in problem.getBackwardsSuccessors(state):
                if (succ[0], initialGoal) not in closedBackwardSet:
                    succState, succAction, succCost = succ
                    bValue = 2 * (pathCost + succCost) + backwardsHeuristic(succState, problem) - heuristic(succState, problem)
                    newNode = (succState, succAction, pathCost + succCost, path + [(state, action)], initialGoal)
                    openBackward.push(newNode, bValue)
                    if (succState, initialGoal) in openBackwardStates.keys():
                        openBackwardStates[(succState, initialGoal)] = openBackwardStates[(succState, initialGoal)] + 1
                    else:
                        openBackwardStates[(succState, initialGoal)] = 1
                    if (succState, initialGoal) in costBackward.keys() and costBackward[(succState, initialGoal)][0] > (succCost + pathCost):
                        pathBackward[(succState, initialGoal)] = newNode[3]
                        costBackward[(succState, initialGoal)] = [succCost + pathCost, succAction]
                    elif (succState, initialGoal) not in costBackward.keys():
                        pathBackward[(succState, initialGoal)] = newNode[3]
                        costBackward[(succState, initialGoal)] = [succCost + pathCost, succAction]
        # Flip the search direction
        searchDir.switchDir()
    return plan

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

ehc = enforcedHillClimbing
bae = bidirectionalAStarEnhanced
