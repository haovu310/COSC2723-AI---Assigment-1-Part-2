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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        This evaluation function assesses the desirability of taking a given action from the current game state.
        The agent is reflex-based: it looks only one move ahead and makes a decision based on that.

        Parameters:
            - currentGameState: the current state of the game (Pacman position, food, ghost states, etc.)
            - action: the potential move Pacman might take (e.g., 'North', 'South', 'Stop')

        Returns:
            - A numerical score that estimates how good this action is. Higher = better.
        """

        # Generate the state that would result from taking the given action
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Extract Pacman's new position after the move
        newPos = successorGameState.getPacmanPosition()

        # Extract the grid of remaining food
        newFood = successorGameState.getFood()

        # Extract information about all ghosts after the move
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Start with the successor state's base score
        # This already includes the standard game scoring (e.g., eating food)
        score = successorGameState.getScore()

        # (1) Encourage Pacman to move closer to food
        foodList = newFood.asList()  # Convert food grid into list of coordinates
        if foodList:
            # Find the Manhattan distance to the closest food
            closestFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
            # The closer the food, the higher the score bonus
            score += 10 / (closestFoodDist + 1)  # +1 to avoid division by zero
        else:
            # If there is no food left, it a winning situation, then we will boost the score
            score += 100

        # (2) Penalize proximity to dangerous ghosts
        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            dist = manhattanDistance(newPos, ghostPos)

            if newScaredTimes[i] == 0:
                # Ghost is active (dangerous)
                if dist <= 1:
                    # Very close ghost — high danger! Strong penalty
                    score -= 500
                else:
                    # Further ghosts — apply smaller penalty
                    score -= 2 / dist
            else:
                # Ghost is scared — we may want to chase it for points
                score += 5 / (dist + 1)  # Closer scared ghosts are better

        # (3) Discourage Pacman from stopping
        if action == Directions.STOP:
            score -= 10  # Standing still is rarely helpful in Pacman

        return score


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
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Pacman is agentIndex 0 (MAX), ghosts are agentIndex 1 to N-1 (MIN layers).
        A single ply = 1 Pacman move + moves for each ghost.
        """

        def minimax(state, depth, agentIndex):
            """
            Recursive minimax function that returns a numerical score
            for the given state, from the perspective of the current agent.
            """
            # Base case: if game is over or max depth reached, evaluate the state
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # MAX player (Pacman)
            if agentIndex == 0:
                # Pacman tries to maximize the score
                maxValue = float('-inf')

                for action in state.getLegalActions(agentIndex):
                    # Generate successor state from taking action
                    successor = state.generateSuccessor(agentIndex, action)

                    # Recurse to the next agent (first ghost)
                    value = minimax(successor, depth, agentIndex + 1)

                    # Track the best value seen so far
                    maxValue = max(maxValue, value)

                return maxValue

            else:
                # MIN player (ghost)
                minValue = float('inf')

                # Determine which agent comes next
                nextAgent = agentIndex + 1
                nextDepth = depth

                # If we've reached the last agent, wrap around to Pacman
                # and reduce depth by 1 (1 full ply complete)
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth -= 1

                for action in state.getLegalActions(agentIndex):
                    # Generate successor state from ghost action
                    successor = state.generateSuccessor(agentIndex, action)

                    # Recurse to the next agent or next ply
                    value = minimax(successor, nextDepth, nextAgent)

                    # Track the worst-case scenario (ghosts minimize)
                    minValue = min(minValue, value)

                return minValue

        # Top-level decision for Pacman
        bestScore = float('-inf')  # Highest value seen so far
        bestAction = None  # Action that gives the best value

        # Evaluate all possible Pacman moves (legal actions at root)
        for action in gameState.getLegalActions(0):
            # Generate successor state after Pacman takes the action
            successor = gameState.generateSuccessor(0, action)

            # Evaluate that state using minimax starting with first ghost
            score = minimax(successor, self.depth, 1)

            # Update best action if this move yields a better score
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction, with alpha-beta pruning to improve efficiency.

        Pacman is agentIndex 0 (MAX), ghosts are agentIndex 1 to N-1 (MIN layers).
        A single ply = 1 Pacman move + moves for each ghost.
        """

        def minimax(state, depth, agentIndex, alpha, beta):
            """
            Recursive minimax function with alpha-beta pruning that returns a numerical score
            for the given state, from the perspective of the current agent.
            """
            # Base case: if game is over or max depth reached, evaluate the state
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # MAX player (Pacman)
            if agentIndex == 0:
                # Pacman tries to maximize the score
                maxValue = float('-inf')

                for action in state.getLegalActions(agentIndex):
                    # Generate successor state from taking action
                    successor = state.generateSuccessor(agentIndex, action)

                    # Recurse to the next agent (first ghost)
                    value = minimax(successor, depth, agentIndex + 1, alpha, beta)

                    # Track the best value seen so far
                    maxValue = max(maxValue, value)

                    # Alpha-beta pruning: update alpha
                    alpha = max(alpha, maxValue)
                    if maxValue >= beta:
                        break  # Prune remaining branches

                return maxValue

            else:
                # MIN player (ghost)
                minValue = float('inf')

                # Determine which agent comes next
                nextAgent = agentIndex + 1
                nextDepth = depth

                # If we've reached the last agent, wrap around to Pacman
                # and reduce depth by 1 (1 full ply complete)
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth -= 1

                for action in state.getLegalActions(agentIndex):
                    # Generate successor state from ghost action
                    successor = state.generateSuccessor(agentIndex, action)

                    # Recurse to the next agent or next ply
                    value = minimax(successor, nextDepth, nextAgent, alpha, beta)

                    # Track the worst-case scenario (ghosts minimize)
                    minValue = min(minValue, value)

                    # Alpha-beta pruning: update beta
                    beta = min(beta, minValue)
                    if minValue <= alpha:
                        break  # Prune remaining branches

                return minValue

        # Top-level decision for Pacman
        bestScore = float('-inf')  # Highest value seen so far
        bestAction = None  # Action that gives the best value
        alpha = float('-inf')  # Initialize alpha (max's lower bound)
        beta = float('inf')  # Initialize beta (min's upper bound)

        # Evaluate all possible Pacman moves (legal actions at root)
        for action in gameState.getLegalActions(0):
            # Generate successor state after Pacman takes the action
            successor = gameState.generateSuccessor(0, action)

            # Evaluate that state using minimax starting with first ghost
            score = minimax(successor, self.depth, 1, alpha, beta)

            # Update best action if this move yields a better score
            if score > bestScore:
                bestScore = score
                bestAction = action

            # Update alpha after each action
            alpha = max(alpha, bestScore)

        return bestAction


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
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
