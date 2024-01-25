# inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattanDistance


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        # Get the sum of all key values.
        total = sum(self.values())
        # Normalize the distribution if the sum is not 0, which indicates that at least one key has a value.
        if total != 0:
            # Iterate over all keys, dividing their values by the total.
            for key in self:
                value = self[key]  # obtaining the value related to the current key
                normalized_value = value / total  # calculate the normalized value
                self[key] = normalized_value  # recalculate the value corresponding to the current key.

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        # Create a random value between 0 and the distribution's grand total of values.
        rand_val = random.uniform(0, sum(self.values()))
        # Set up a counter for the value's running total.
        total = 0

        # Iterate the process with the distribution's items.
        for key, value in self.items():
            # Add the value of the current item to the running total
            total = total + value
            # Return the key if the random value is inside the current item's range of values.
            if rand_val < total:
                return key

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        if ghostPosition == jailPosition:
            # If ghost is in jail, the distance should be None
            return float(noisyDistance is None)
        else:
            # If ghost is not in jail, calculate the true distance and return the observation probability
            if noisyDistance is not None:  # If noisy distance is not None
                trueDistance = manhattanDistance(pacmanPosition, ghostPosition)  # Calculate true distance
                observationProb = busters.getObservationProbability(noisyDistance, trueDistance)  # Calculate observation probability
            else:  # If noisy distance is None
                observationProb = 0.0  # Set observation probability to 0
            return observationProb  # Return the observation probability

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        probability_distribution = self.beliefs

        # Get Pacman and jail positions
        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()

        # Create an empty new probability distribution
        new_probability_distribution = DiscreteDistribution()

        # Update new probability distribution for each possible ghost position
        for ghostPosition in self.allPositions:
            # Calculate the probability of the observation given the locations of Pacman and the ghost.
            observationProb = self.getObservationProb(observation, pacmanPosition, ghostPosition, jailPosition)
            # Update the probability that the ghost is in its current location using the observation probability and the previous belief as a product.
            new_probability_distribution[ghostPosition] = observationProb * probability_distribution[ghostPosition]

        self.beliefs = new_probability_distribution

        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        # Initialize a new DiscreteDistribution for the updated beliefs
        new_probability_distribution = DiscreteDistribution()

        # For each possible ghost position in the current state
        for oldPos in self.allPositions:
            # Using the old position and the current game state, determine the probability distribution over new locations.
            newPosDist = self.getPositionDistribution(gameState, oldPos)

            # Update the new belief for each potential new location and its probability in the distribution.
            for newPos, prob in newPosDist.items():
                new_probability_distribution[newPos] += self.beliefs[oldPos] * prob

        # Update the current beliefs to the new beliefs
        self.beliefs = new_probability_distribution

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        # Calculate the number of positions on the board
        numPositions = len(self.legalPositions)

        # Calculate the number of particles that should be assigned to each position
        particlesPerPosition = self.numParticles // numPositions

        # Loop over each position and add particles to the particle list
        for position in self.legalPositions:
            for i in range(particlesPerPosition):
                self.particles.append(position)

        # Assign any remaining particles to extra randomly selected positions
        remainingParticles = self.numParticles % numPositions
        if remainingParticles > 0:
            extraPositions = random.sample(self.legalPositions, remainingParticles)
            for pos in extraPositions:
                self.particles.append(pos)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()

        # Create a new distribution to represent the updated beliefs
        newBeliefDist = DiscreteDistribution()

        # Determine the probability that each particle will provide the specified observation given its position and the location of the prison for that particle.
        for particle in self.particles:
            probability = self.getObservationProb(observation, pacmanPosition, particle, jailPosition)
            # Add the probability to the new belief distribution dictionary
            newBeliefDist[particle] = newBeliefDist.get(particle, 0) + probability

        # Check if all the weights are zero
        if newBeliefDist.total() == 0:
            self.initializeUniformly(gameState)
        else:
            # Normalize the distribution to ensure that the probabilities sum to 1
            newBeliefDist.normalize()

            # Update the list of particles based on the new distribution
            self.particles = []
            for i in range(self.numParticles):
                particle = newBeliefDist.sample()
                self.particles.append(particle)

        return newBeliefDist

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        newParticles = []
        for oldPos in self.particles:
            # get the probability distribution of new positions given the current position
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            # sample a new position from the probability distribution
            newParticle = newPosDist.sample()
            # add the new position to the list of new particles
            newParticles.append(newParticle)
            # update the list of particles with the new particles
        self.particles = newParticles

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        beliefDist = {}
        # iterate over all particles
        for particle in self.particles:
            # increment particle count
            beliefDist[particle] = beliefDist.get(particle, 0) + 1
            # compute total number of particles
        totalParticles = len(self.particles)
        # iterate over all positions in beliefDist
        for key in beliefDist:
            # divide particle count by total particles to get probability
            beliefDist[key] = beliefDist[key] / totalParticles
        return DiscreteDistribution(beliefDist)

class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        # Generate permutations of legal positions
        permutations = list(itertools.product(self.legalPositions, repeat=self.numGhosts))

        # Get the number of permutations and particles
        numPermutations = len(permutations)
        numParticles = self.numParticles

        # Clear existing particles
        self.particles = []

        # Add complete sets of permutations
        for i in range(numParticles // numPermutations):
            self.particles.extend(permutations)

        # Add remaining permutations
        remainingParticles = numParticles % numPermutations
        if remainingParticles > 0:
            self.particles.extend(permutations[:remainingParticles])

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        pacmanPosition = gameState.getPacmanPosition()
        new_probability_distribution = DiscreteDistribution()

        # Calculate probability of each particle being correct
        for particle in self.particles:
            particleProb = 1.0
            for ghostIndex, ghostPosition in enumerate(particle):
                observationProb = self.getObservationProb(observation[ghostIndex], pacmanPosition, ghostPosition, self.getJailPosition(ghostIndex))
                particleProb = particleProb * observationProb
            new_probability_distribution[particle] = new_probability_distribution[particle] + particleProb

        # Handle the case where all particles receive zero weight
        if new_probability_distribution.total() == 0:
            self.initializeUniformly(gameState)
        else:
            # Normalize the distribution and resample particles
            new_probability_distribution.normalize()
            self.particles = []
            for i in range(self.numParticles):
                self.particles.append(new_probability_distribution.sample())

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            prevGhostPositions = newParticle

            # iterate through each ghost
            for i in range(self.numGhosts):
                # get the distribution of possible positions for this ghost
                newPosDist = self.getPositionDistribution(gameState, prevGhostPositions, i, self.ghostAgents[i])
                # sample a new position for this ghost
                newGhostPos = newPosDist.sample()
                # update the new particle with the new position for this ghost
                newParticle[i] = newGhostPos
            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
