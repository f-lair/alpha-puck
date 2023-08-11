# AlphaPuck (DecQN)

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
[![CI](https://github.com/f-lair/alpha-puck/actions/workflows/ci.yml/badge.svg)](https://github.com/f-lair/alpha-puck/actions/workflows/ci.yml)
![License](https://img.shields.io/github/license/f-lair/alpha-puck)

This work was done as part of the Reinforcement Learning (RL) lecture at the University of Tübingen in the summer semester of 2023.
The goal was to develop an RL agent for a 2D two-player hockey game that can beat two PD-controlled basic opponent players.
There was also a final tournament against all RL agents developed by the other participants in the lecture.

We implemented the Decoupled Q-Networks (DecQN) [[1]](#1) algorithm, an adaption of classical DQN for continuous control problems.
To allow training against a more diverse set of opponents and therefore guide to a more robust playing style, we also used a simplified version of league training [[2]](#2).

## Installation

To run the scripts in this repository, **Python 3.10** is needed.
Then, simply create a virtual environment and install the required packages via

```bash
pip install -r requirements.txt
```

## Usage

The same python script `src/main.py` is used for both training and evaluating an RL agent.
It implements a command line interface that allows choosing between these two tasks through subcommands `train` and `test`.
For the details, invoke the script with the flag `-h`/`--help`.


## References

<a id="1">[1]</a> 
Tim Seyde et al. “Solving Continuous Control via Q-learning”. 
In: The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. 
URL: https://openreview.net/pdf?id=U5XOGxAgccS.

<a id="2">[2]</a> 
Oriol Vinyals et al. “Grandmaster level in StarCraft II using multi-agent reinforcement learning”.
In: Nature 575.7782 (Nov. 2019), pp. 350–354. 
URL: https://doi.org/10.1038/s41586-019-1724-z.
