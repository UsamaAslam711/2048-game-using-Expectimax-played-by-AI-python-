# 2048-game-using-Expectimax-played-by-AI-python-
This is a 2048 game using expecti-max algorithm. Language used for coding is python. 


One of the most addictive game is 2048 which is available across all platforms even in wearable devices. We created an agent which capable of playing the game without human interference and with an increased possibility of winning this game when compared to an average player. For this, we used exceptiminmax often referred as expectimax to solve the game which calculates all the possible moves and selects the one with highest probability.

Proposed Approach:The proposed approach to solve this
project is by using Q-Learning, the agent travels to all the
possible nodes and stores the rewards for each and every
move and uses Q-Learning to make the agent to decide
what is the best action to take which helps the agent by
providing all the required information about the path to take
in and win the game.
With this approach we faced a setback when we are
expanding the nodes since we have a very large state space
the Q-Learning helps the agent to take the moves initially
but as we go deep in the tree the the exploration algorithm
cannot handle the large state space and runs out of memory
so we decided to change the approach.
Used Approach:Since we had a problem in the node
exploration due to the enormously large state space so we
have to use an approach that helps that agent not to go and
stuck in the depth. So to solve the puzzle we use
Expectimax which helps the agent to choose the best path
and thereby win the game.
Expectimax:Expectimax is a special variation of minimax
game tree used to play two-player zero-sum games such as
backgammon by artificial intelligence systems , in which
the moves depend upon the player's skill and random
chances. Expectimax has chance nodes in addition to min
and max, which takes the expected value of random event
that is about to occur, While in a a normal minimax
approach we have min and max nodes.
Expectimax's chance nodes are interleaved with max and
min nodes, in contrast with minimax where the levels of the
tree alternate from max to min until the depth limit of the
tree has been reached, where interleaving depends on the
game. Chance nodes take a weighted average where weight
is the probability that the child is reached instead of taking
the min or max of utility values. In each turn the game is
evaluated as the max node and the opponent turn is
calculated as the min node.

