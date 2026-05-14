from dataclasses import dataclass
from itertools import count

import heapq
import sys


State = frozenset


@dataclass(frozen=True)
class Action:
    name: str
    args: tuple
    precond: frozenset
    add: frozenset
    delete: frozenset

    def __str__(self):
        return f"{self.name}({', '.join(self.args)})"

    def applicable(self, state):
        return self.precond <= state

    def apply(self, state):
        return (state - self.delete) | self.add


def ground_actions(blocks):
    actions = []
    for x in blocks:
        actions.append(Action(
            name="pickup", args=(x,),
            precond=frozenset({("ontable", x), ("clear", x), ("armempty",)}),
            add=frozenset({("holding", x)}),
            delete=frozenset({("ontable", x), ("clear", x), ("armempty",)}),
        ))
        actions.append(Action(
            name="putdown", args=(x,),
            precond=frozenset({("holding", x)}),
            add=frozenset({("ontable", x), ("clear", x), ("armempty",)}),
            delete=frozenset({("holding", x)}),
        ))
        for y in blocks:
            if x == y:
                continue
            actions.append(Action(
                name="stack", args=(x, y),
                precond=frozenset({("holding", x), ("clear", y)}),
                add=frozenset({("on", x, y), ("clear", x), ("armempty",)}),
                delete=frozenset({("holding", x), ("clear", y)}),
            ))
            actions.append(Action(
                name="unstack", args=(x, y),
                precond=frozenset({("on", x, y), ("clear", x), ("armempty",)}),
                add=frozenset({("holding", x), ("clear", y)}),
                delete=frozenset({("on", x, y), ("clear", x), ("armempty",)}),
            ))
    return actions


def applicable_actions(state, actions):
    return [a for a in actions if a.applicable(state)]


def successors(state, actions):
    for a in actions:
        if a.applicable(state):
            yield a, a.apply(state)


def goal_reached(state, goal):
    return goal <= state


def subgoals_reached(state, goal):
    subgoals_reached = 0

    for subgoal in goal:
        if subgoal in state:
            subgoals_reached += 1

    return subgoals_reached / len(goal)


def fmt_state(state):
    return sorted(" ".join(map(str, lit)) for lit in state)


def find_plan(initial, goal, actions):
    plans = []

    # We use a heap queue to sort the queue by a scoring heuristic,
    # and we use a counter to break ties.
    counter = count()
    heapq.heappush(plans, (0, next(counter), []))

    visited_states = []

    while len(plans) > 0:
        _, _, plan = heapq.heappop(plans)

        # Find the state for the given plan
        state = initial
        for a in plan:
            state = a.apply(state)

        # Push back to the queue all new possible plans
        for a in applicable_actions(state, actions):
            new_state = a.apply(state)

            # Skip the state if we have already seen it
            if new_state in visited_states:
                continue

            # Compute the path that lead to the state
            new_plan = plan + [a]

            # Finish if the new_state is the goal
            if goal_reached(new_state, goal):
                return new_plan

            # Push the plan back to the queue depending on its score
            score = subgoals_reached(new_state, goal)
            heapq.heappush(plans, (-score, next(counter), new_plan))

        visited_states.append(state)

    return None




if __name__ == "__main__":
    n = int(sys.argv[1])
    blocks = list(range(1,n+1))

    initial = {
        ("ontable", 1),
        ("clear", n),
        ("armempty",),
    }
    goal = {
        ("ontable", n),
        ("clear", 1),
        ("armempty",),
    }

    for i in range(1,n):
        initial.add(("on", i+1, i))
        goal.add(("on", i, i+1))

    initial = frozenset(initial)
    goal = frozenset(goal)

    actions = ground_actions(blocks)

    # print(f"Blocks: {blocks}")
    # print(f"Grounded actions: {len(actions)}")
    # print("Initial state:")
    # for line in fmt_state(initial):
    #     print(f"  {line}")
    # print("Goal:")
    # for line in fmt_state(goal):
    #     print(f"  {line}")
    # print(f"Goal already reached? {goal_reached(initial, goal)}")
    # print("Applicable actions in initial state:")
    # for a in applicable_actions(initial, actions):
    #     print(f"  {a}")

    plan = find_plan(initial, goal, actions)
    if plan is None:
        print("No plan reaches the goal from the initial state.")
    else:
        print("We found a plan.")
        # print("The plan is:")
        # for a in plan:
        #     print(f"  {a}")
