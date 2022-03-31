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
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        # Manhattan distance to every food:
        newFoodList = newFood.asList()
        foodDist = []
        for foodPos in newFoodList:
            foodDist.append(util.manhattanDistance(newPos, foodPos))

        # Manhattan distance to closest food:
        closestFood = (sum(foodDist)/len(foodDist) if foodDist else 1)

        # Manhattan distance to ghosts:
        ghostDist = [util.manhattanDistance(
            newPos, ghostState.getPosition()) for ghostState in newGhostStates]

        # Furthest Ghost:
        furthestGhost = max(ghostDist, default=1)

        return score + furthestGhost/closestFood


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        def value(state, agent, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None
            if agent == 0:
                return maxValue(state, agent, depth)
            else:
                return minValue(state, agent, depth)

        def maxValue(state, agent, depth):
            v = float('-inf')
            bestAction = None
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                result = value(successor, 1, depth)
                if v != max(v, result[0]):
                    bestAction = action
                v = max(v, result[0])
            return v, bestAction

        def minValue(state, agent, depth):
            v = float('inf')
            bestAction = None
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                nextAgent = agent + 1
                nextDepth = depth
                if nextAgent >= state.getNumAgents():
                    nextAgent = 0
                    nextDepth = depth + 1
                result = value(successor, nextAgent, nextDepth)
                if v != min(v, result[0]):
                    bestAction = action
                v = min(v, result[0])
            return v, bestAction

        res = value(gameState, 0, 0)

        return res[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(state, agent, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None
            if agent == 0:
                return maxValue(state, agent, depth, alpha, beta)
            else:
                return minValue(state, agent, depth, alpha, beta)

        def maxValue(state, agent, depth, alpha, beta):
            v = float('-inf')
            bestAction = None
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                result = value(successor, 1, depth, alpha, beta)
                if v != max(v, result[0]):
                    bestAction = action
                    v = result[0]
                if v > beta:
                    return v, bestAction
                alpha = max(alpha, v)
            return v, bestAction

        def minValue(state, agent, depth, alpha, beta):
            v = float('inf')
            bestAction = None

            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                nextAgent, nextDepth = agent + 1, depth
                if nextAgent >= state.getNumAgents():
                    nextAgent, nextDepth = 0, depth + 1

                result = value(successor, nextAgent, nextDepth, alpha, beta)
                if v != min(v, result[0]):
                    bestAction = action
                    v = result[0]
                if v < alpha:
                    return v, bestAction
                beta = min(beta, v)
            return v, bestAction

        res = value(gameState, 0, 0, float('-inf'), float('inf'))

        return res[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(state, agent, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None
            if agent == 0:
                return maxValue(state, agent, depth)
            else:
                return expectValue(state, agent, depth)

        def maxValue(state, agent, depth):
            v = float('-inf')
            bestAction = None
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                result = value(successor, 1, depth)
                if v != max(v, result[0]):
                    bestAction = action
                v = max(v, result[0])
            return v, bestAction

        def expectValue(state, agent, depth):
            bestAction = None
            sum = 0
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                nextAgent = agent + 1
                nextDepth = depth
                if nextAgent >= state.getNumAgents():
                    nextAgent = 0
                    nextDepth = depth + 1
                result = value(successor, nextAgent, nextDepth)
                sum += result[0]
            return sum, bestAction

        res = value(gameState, 0, 0)

        return res[1]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    notNewGhostStates = currentGameState.getGhostStates()
    notNewScaredTimes = [
        ghostState.scaredTimer for ghostState in notNewGhostStates]

    # Get the manhattan distances of food and ghost

    # Manhattan distance to every food:
    newFood = currentGameState.getFood()
    newPos = currentGameState.getPacmanPosition()
    newFoodAsList = newFood.asList()
    foodManhattanDistances = []
    for foodPosition in newFoodAsList:
        foodManhattanDistances.append(
            util.manhattanDistance(newPos, foodPosition))

    # Closest Food:
    closestFood = (sum(foodManhattanDistances) /
                   len(foodManhattanDistances) if foodManhattanDistances else 1)

    # Manhattan distance to ghosts:
    distanceGhost = [manhattanDistance(
        newPos, ghostState.getPosition()) for ghostState in notNewGhostStates]

    # Sum of Ghost distances:
    furthestGhost = sum(distanceGhost)

    for i in range(len(distanceGhost)):
        if distanceGhost[i] < notNewScaredTimes[i]:
            return score + (notNewScaredTimes[i]-distanceGhost[i])/closestFood

    return score + furthestGhost/closestFood


# Abbreviation
better = betterEvaluationFunction
