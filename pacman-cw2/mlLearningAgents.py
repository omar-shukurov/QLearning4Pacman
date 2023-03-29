# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        # This variable will just pass the state when needed
        self.currentState = state
        



class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        #Class variables that will help with the Q learning implementation
        self.prevState = None
        self.prevAction = None
        self.qValues = util.Counter()
        self.count = util.Counter()

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        #Calculates the reward by taking the change in score
        #Eating food increases score, other wise by default it decreases
        return endState.getScore() - startState.getScore()



    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        #From the stored q values return the values for the given state/action pair, Q(s,a)
        values = self.qValues[(state.currentState, action)]
        return values



    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        #Get all the possible legal moves pacman can make
        legal = state.currentState.getLegalPacmanActions()
        if not legal:
            return 0
        legal.remove('Stop')

        #Then loops through all Q(s,a) pairs and adds them to the list
        qList = []
        for n in legal:
            q = self.getQValue(state, n)
            qList.append(q)
        
        #Check if list is empty, if it is empty return 0 otherwise return the highest q value in the list
        if not qList:
            return 0
        
        return max(qList)



    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        #Get the values that will be required for the q learning update
        initialQ = self.getQValue(state,action)
        maxQ = self.maxQValue(nextState)

        #Update the Q value using the formula Q(s,a) <- Q(s,a) + alpha(R(s) + gamma * max(Q(s',a')) - Q(s,a))
        newQ = initialQ + self.getAlpha() * (reward + self.getGamma() * maxQ - initialQ)

        self.qValues[(state.currentState, action)] = newQ # Update the Q value for the previous state


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        #Get the dictionary that keeps count and then increment the given state,action pair
        values = self.count
        values[(state, action)] += 1


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        #Fetches the count for given state,action pair 
        values = self.count
        return values[(state, action)]


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        #Exploration function which tires to makes agent try different path
        #We return estimated reward if that state action has not been done enough
        #Else if we explored that state/action enough we return it's utility

        rewardPlus = 2 #Estimated reward
        numMin = 1 #Minimum amount of time to try a given state and action

        if counts < numMin:
            return rewardPlus
        else:
            return utility


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # logging to help you understand the inputs, feel free to remove

        stateFeatures = GameStateFeatures(state)
        
        #First check that there is a previous state/action pair, if None then initialise them
        #Make the state the first state of game, and the action one the the legal actions
        if self.prevState == None:
            randomAction = random.choice(legal)
            self.prevAction = randomAction
            self.prevState = state
            #If there was no previous action then just perform random legal action
            return randomAction

        oldStateFeatures = GameStateFeatures(self.prevState) #Last state we were in
        
        #Set our own epsilon value (Very low because we also use exploration function scroll down)
        if self.getAlpha() == 0:
            self.epsilon = 0 #Turn it to 0 when training is done
        else:
            self.epsilon = 0.05

        # This is the implementation for e greedy 
        if util.flipCoin(self.epsilon):
            return  random.choice(legal)
        
        #Calculate variables that will be required
        prevState = self.prevState
        reward = self.computeReward(prevState, state)
        prevAction = self.prevAction

        #Responsible for the learning (qvalue updates)
        self.learn(oldStateFeatures, prevAction, reward, stateFeatures)
        
        #To pick the action with the highest qValue (Or explore using explore function)
        move = legal[0]
        hold = []
        maxQ = -9999
        for n in legal:
            q = self.getQValue(stateFeatures,n)

            #Exploration function called, to try explore what has not been explored
            q = self.explorationFn(q, self.count[(state, n)])

            hold.append((n,q))
            if q > maxQ:
                maxQ = q
                move = n

        # Update the previous values
        self.prevAction = move
        self.prevState = state
        self.updateCount(state,move)
        return move


    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        #Perform one last update which is influenced by game outcome
        reward = self.computeReward(self.prevState, state) 
        stateFeatures = GameStateFeatures(state)
        oldStateFeatures = GameStateFeatures(self.prevState)
        self.learn(oldStateFeatures, self.prevAction, reward, stateFeatures)

        #Reset the given variables
        self.prevAction = None
        self.prevState = None
        
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
