# Example: Chained Spring-Mass System using MultiMechanicalSystem
# This script demonstrates a system of 3 masses connected by springs in a chain.

import sys
sys.path.append('.')
from source import MultiMechanicalSystem

# Masses, dampings, and spring constants for 3 masses
masses = [1.0, 1.0, 1.0]
dampings = [0.1, 0.1, 0.1]
spring_constants = [2.0, 2.0, 2.0]

# Initial positions and velocities
initial_positions = [1.0, 0.0, -1.0]
initial_velocities = [0.0, 0.0, 0.0]

# Couplings: (i, j) means spring between mass i and mass j
couplings = [(0, 1), (1, 2)]
coupling_constants = [1.0, 1.0]

# Create the system
system = MultiMechanicalSystem.MultiMechanicalSystem(
    masses, dampings, spring_constants, initial_positions, initial_velocities, couplings, coupling_constants
)

# Simulate and plot
system.simulate(T=10.0, h=0.01)
system.createPlot()
