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
        states = self.mdp.getStates()
        numIter = self.iterations
        while (numIter > 0):
          newV = util.Counter()
          for state in states:
            opAct = self.computeActionFromValues(state)
            opVal = self.computeQValueFromValues(state, opAct)
            newV[state] = opVal
          for state in states:
            self.values[state] = newV[state]
          numIter -= 1


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
        if (self.mdp.isTerminal(state)):
          return 0
        tStateAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        qVal = 0
        for t in tStateAndProbs:
          reward = self.mdp.getReward(state, action, t[0])
          qVal += t[1]*(self.mdp.getReward(state, action, t[0]) + self.discount*self.getValue(t[0]))
        return qVal


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxiValue = -float('inf')
        optiPolicy = None
        actions = self.mdp.getPossibleActions(state)
        '''print('state')
        print(state)
        print('action')
        print(actions)'''
        for action in actions:
          tempVal = self.computeQValueFromValues(state, action)
          if (tempVal > maxiValue):
            maxiValue = tempVal
            optiPolicy = action
        return optiPolicy

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
        states = self.mdp.getStates()
        numIter = self.iterations
        i = 0
        while (numIter > 0):
          vState = states[i]
          opAct = self.computeActionFromValues(vState)
          opVal = self.computeQValueFromValues(vState, opAct)
          self.values[vState] = opVal
          i += 1
          i = i % len(states)
          numIter -= 1

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
        states = self.mdp.getStates()
        predecessors = set()

        for state in states:
          self.computePredeccessors(state, predecessors)

        queue = util.PriorityQueue()
        for state in states:
          if (not self.mdp.isTerminal(state)):
            optimalAction = self.computeActionFromValues(state)
            optimalQ = self.computeQValueFromValues(state, optimalAction)
            diff = abs(self.values[state] - optimalQ)
            queue.push(state, -diff)

        for i in range(self.iterations):
          if queue.isEmpty():
            return
          popState = queue.pop()
          optiAct = self.computeActionFromValues(popState)
          optiQ = self.computeQValueFromValues(popState, optiAct)
          self.values[popState] = optiQ
          for pred in predecessors:
            if (pred[0] == popState):
              temp = abs(self.values[pred[2]] - self.computeQValueFromValues(pred[2], self.computeActionFromValues(pred[2])))
              if temp > self.theta:
                queue.update(pred[2], -temp)

    def computePredeccessors(self, state, pred):
      actions = self.mdp.getPossibleActions(state)
      for action in actions:
        tStates = self.mdp.getTransitionStatesAndProbs(state, action)
        for t in tStates:
          if (t[1] != 0):
            pred.add((t[0], action, state))


