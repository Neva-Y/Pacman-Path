# Student Information

**Course:** COMP90054 AI Planning for Autonomy

**Semester:** Semester 1, 2023


**Student: Xing Yang Goh - 1001969 - xingyangg**

your full name - your student number - your canvas student login

> Replace the lines above with your correct details. Your student number should only be the **numbers**. For example: Guang Hu - 000000 - ghu1. 

**Collaborated With:**

> If you worked with another student, please include their **full name** and ask them to provide you with the **url to their github codebase**. Their codebase should be private, you will not have access to the code, so there's no issue with knowing their URL, e.g. Collaborated with: Lionel Messi - URL: github.com/what-a-worldcup.

# Self Evaluation

>**Note**
> Do not exceed 500 words for each Part. This is indicative, no need to have 500 words, and it's not a strict limit.

## Part 1
#### Self Evaluated Marks (3 marks):
3
> Please replace the above 0 with the mark you think you earned for this part. Consider how many (yours/ours) tests pass, the quality of your code, what you learnt, and [mainly for the final task] the quality of the tests that you wrote
#### Code Performance
This graph is just an example of how you can include your plots in markdown.

> Please explain the code performance of your solution. You can create a video, include figures, tables, etc. Make sure to complement them with text explaining the performance.

#### Learning and Challenges
> Please include your top lessons learnt, and challenges faced.  
- Exiting the EHC after expanding all the child nodes, using a priority queue to expand the child with the lowest heuristic
- Checking if any of the expanded nodes were the goal nodes instead of checking after popping the node from the queue, led to less expanded nodes compared to the expected solution

#### Ideas That Almost Worked Well
> If you tried ideas that did not make it to the final code, please include them here and explain why they didn't make it.
- Tried to perform the EHC search in a single function, but using a helper Improve function aided implementation 
#### Justification
> Please state the reason why you have assigned yourself these marks.
- Passed all the test cases for this search algorithm and followed the implementation shown in the lecture

#### New Tests Shared @ ED

> Tell us about your testcases and why were they useful

## Part 2
#### Self Evaluated Marks (3 marks):
3
> Please replace the above 0 with the mark you think you earned for this part. Consider how many (yours/ours) tests pass, the quality of your code, what you learnt, and [mainly for the final task] the quality of the tests that you wrote.
#### Code Performance
> Please explain the code performance of your solution. You can create a video, include figures, tables, etc. Make sure to complement them with text explaining the performance.

#### Learning and Challenges
> Please include your top lessons learnt, and challenges faced.  
- Had to flip between forward and backward searches, converging to a minimum cost path when these searches have overlapping states since the manhattan distance heuristic is always admissible. Performed this directional flipping by creating a SearchDir class that initialises to 'Forward' and a method that flips this direction to 'Backwards' at the end of the while loop
- Difficulty in keeping track of states in the priority queue since there could be duplicate states if they have not been popped yet (and there isn't an easy way to acquire the states in the heap), ended up using a dictionary to keep track of these items with a counter
- General learning the set notation and reading the paper to understand the pseudocode.
- Implementing the best data structures for the tasks, such as dictionaries, PQs and sets for quick lookup and popping
#### Ideas That Almost Worked Well
- Using a set to keep track of items in the PQ, but this encounters issues with duplicate states in the heap. Performed a check to see if the item is in the set before adding to the PQ, which vastly reduces the number of node expansions. However, this method assumes that duplicate states that are already in the PQ will never form the ideal path.
> If you tried ideas that did not make it to the final code, please include them here and explain why they didn't make it.

#### New Tests Shared @ ED

> Tell us about your testcases and why were they useful

#### Justification
- Passed all the test cases for the Bi-Directional A* algorithm
- Utilised efficient data structures with sets and dictionaries to support the membership checks and acquiring minimum b-value nodes from the PQ
> Please state the reason why you have assigned yourself these marks.

## Part 3
#### Self Evaluated Marks (4 marks):
4
> Please replace the above 0 with the mark you think you earned for this part. Consider how many (yours/ours) tests pass, the quality of your code, what you learnt, and [mainly for the final task] the quality of the tests that you wrote
#### Code Performance
> Please explain the code performance of your solution. You can create a video, include figures, tables, etc. Make sure to complement them with text explaining the performance.

#### Learning and Challenges
> Please include your top lessons learnt, and challenges faced.  
- Took a long time to internalise the problem, since now having the coordinates matching isn't enough to form an optimal path since the agent has to eat all the food instead of a single goal coordinate
- Encoded this information by adding the grid state in the nodes, where the forward search begins with all the food states on the grid set to true and the backwards search from the goal state setting all other goal states to False. Whenever the forward search traverses across a food state it sets it to false, while the backward search sets food states it traverses across to true. This means that when the coordinates + the grid states match up from the backwards and forwards search, the ideal path is formed for an admissible heuristic.
- Tried to create a heuristic that used minimum/maximum manhattan distance between the state and the closest/furthest food node which had some success in reducing node expansions, however, this heuristic cannot be proven to be admissible due to the nature of having multiple food states, therefore, a admissible heuristic of 0 was used to ensure an optimal path is found every time.
- The multiple food nodes led to states in the backwards search that were identical from different starting food nodes, which were not considered as they have been added to the closedBackward Set. This was problematic since these states from different starting food nodes in the backwards should be considered (as the food node that the backward search starts from is where the path will end, and for most problems there could be only one ending food state that produces an optimal solution)
- To overcome this "multiple unique food states in backwards search" problem, I added the initial food node coordinates into the tuple for the sets and dictionaries

#### Ideas That Almost Worked Well
> If you tried ideas that did not make it to the final code, please include them here and explain why they didn't make it.
- Creating multiple heaps for the open backwards PQ, one for each goal node and looped through it, this did not lead to correct solutions as we are no longer popping the minimum priority node so a non-optimal solution can be found before an optimal one

#### Justification
> Please state the reason why you have assigned yourself these marks.
- Passed all the test cases for the Bi-Directional A* algorithm with multiple food states
- Utilised efficient data structures with sets and dictionaries, adding information to encode which food states they are from to support multiple food states, while keeping it abstract enough to support a single goal state problem (task 2)

#### New Tests Shared @ ED

> Tell us about your testcases and why were they useful