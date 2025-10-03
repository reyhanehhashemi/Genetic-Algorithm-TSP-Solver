# Genetic-Algorithm-TSP-Solver
Genetic Algorithm implementation to solve the Traveling Salesperson Problem (TSP) using real-world geographic coordinates (Djibouti dataset).
# TSP Genetic Algorithm

A Python implementation of a Genetic Algorithm for solving the Traveling Salesman Problem (TSP).

## Overview

This project provides a complete genetic algorithm framework specifically designed for solving TSP instances. The algorithm includes multiple selection, crossover, and mutation strategies that preserve permutation integrity - essential for TSP solutions.

## Features

### Genetic Algorithm Components
- **Selection Strategies**: Tournament, Roulette Wheel, Rank-based selection
- **Crossover Methods**: One-point, Two-point, Uniform crossover (permutation-preserving)
- **Mutation Operators**: Swap, Inversion, Scramble mutations
- **Customizable Parameters**: Population size, generations, mutation/crossover rates

### TSP-Specific Functionality
- Distance matrix calculation from city coordinates
- Route distance computation with closed tour handling
- Fitness function as inverse of total distance
- Visualization tools for routes and algorithm performance

## Project Structure

```
├── genetic.py          # Core genetic algorithm implementation
├── utils.py            # TSP utilities (distance calc, population generation, plotting)
├── main.py             # Example usage and execution script
├── .gitignore          # Python and development environment ignores
└── data/               # Directory for TSP dataset files
```

## Usage

1. Place TSP dataset files in the `data/` directory
2. Configure parameters in `main.py`:
   - Dataset filename
   - Population size
   - Number of generations
   - Genetic operator rates and strategies
3. Run the algorithm:
   ```bash
   python main.py
   ```

## Key Functions

- `GeneticAlgorithm`: Main algorithm class with configurable operators
- `read_dataset()`: Reads TSP instance files
- `create_distance_matrix()`: Precomputes pairwise distances
- `generate_initial_population()`: Creates random valid tours
- `plot_route()`: Visualizes the best-found route

## Requirements

- Python 3.6+
- matplotlib
- Standard library modules: random, math, typing

## Algorithm Performance

The implementation tracks and visualizes:
- Best fitness history across generations
- Best distance history (actual TSP tour length)
- Final optimized route visualization

This framework provides a flexible foundation for experimenting with different genetic algorithm configurations on various TSP instances.
