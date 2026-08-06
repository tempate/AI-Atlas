# PDDL

Planner: [Fast Downward](https://www.fast-downward.org/).

```sh
cd blocks-world    && python fast-downward.py domain.pddl prob.pddl --search "astar(lmcut())"
cd river-crossing  && python fast-downward.py domain.pddl prob.pddl --search "astar(blind())"
```

The plan is written to `sas_plan` in the current directory.

## Why the different searches

`river-crossing` uses a `forall` precondition, which the translator compiles into
an axiom (`new-axiom@0` in `output.sas`). `lmcut()` does not support axioms and
aborts. Searches that do: `blind()`, `hmax()` (both optimal), `ff()`, `add()`.

State spaces here are tiny, so `blind()` costs nothing.
