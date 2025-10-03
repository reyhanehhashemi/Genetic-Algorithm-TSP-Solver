import random
from typing import List, Callable, Tuple, Optional


class GeneticAlgorithm:
    def __init__(
        self,
        population: List[List[int]],
        fitness_func: Callable[[List[int]], float],
        num_generations: int = 100,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        selection_strategy: str = 'tournament',  # 'tournament', 'roulette', 'rank'
        crossover_strategy: str = 'uniform',  # 'one_point', 'two_point', 'uniform'
        mutation_strategy: str = 'swap',        # 'swap', 'inversion', 'scramble'
        tournament_size: int = 3,
    ) -> None:
        """
        Initialize the genetic algorithm with population and configuration parameters.

        Args:
            population (List[List[int]]): Initial population of individuals.
            fitness_func (Callable[[List[int]], float]): Function to evaluate the fitness of an individual.
            num_generations (int): Number of generations to run the algorithm.
            mutation_rate (float): Probability of mutation for each individual.
            crossover_rate (float): Probability of crossover between pairs.
            selection_strategy (str): Selection method ('tournament', 'roulette', or 'rank').
            crossover_strategy (str): Crossover method ('one_point', 'two_point', or 'uniform').
            mutation_strategy (str): Mutation method ('swap', 'inversion', or 'scramble').
            tournament_size (int): Size of the tournament in tournament selection.
        """
        self.population: List[List[int]] = population
        self.fitness_func: Callable[[List[int]], float] = fitness_func
        self.num_generations: int = num_generations
        self.mutation_rate: float = mutation_rate
        self.crossover_rate: float = crossover_rate
        self.selection_strategy: str = selection_strategy
        self.crossover_strategy: str = crossover_strategy
        self.mutation_strategy: str = mutation_strategy
        self.tournament_size: int = tournament_size

        
        self.best_fitness_history: List[float] = []
        self.best_distance_history: List[float] = []

    def evaluate_population(self) -> List[float]:
            """
            Evaluate the fitness of the current population.

            Returns:
                List[float]: A list containing the fitness value of each individual.
            """
            fitness_values: List[float] = []
            for individual in self.population:
                fitness_values.append(self.fitness_func(individual))
            return fitness_values

    def selection(self, fitnesses: List[float]) -> int:
        """
        Select an individual index based on the configured selection strategy.

        Args:
            fitnesses (List[float]): List of fitness values for the population.

        Returns:
            int: Index of the selected individual.
        """
        population_size = len(self.population)
        if population_size == 0:
            raise ValueError("Population is empty, cannot perform selection.")

        if self.selection_strategy == 'tournament':
            if self.tournament_size <= 0 or self.tournament_size > population_size:
                raise ValueError(f"Tournament size ({self.tournament_size}) must be between 1 and population size ({population_size}).")
            
            best_participant_index = -1
            best_fitness = -float('inf')
            
            
            participant_indices = random.sample(range(population_size), self.tournament_size)
            
            for index in participant_indices:
                if fitnesses[index] > best_fitness:
                    best_fitness = fitnesses[index]
                    best_participant_index = index
            return best_participant_index

        elif self.selection_strategy == 'roulette':
            total_fitness = sum(fitnesses)
            if total_fitness == 0:
                
                return random.randrange(population_size)

            pick = random.uniform(0, total_fitness)
            current_sum = 0
            for i in range(population_size):
                current_sum += fitnesses[i]
                if current_sum >= pick:
                    return i
            
            return population_size - 1 

        elif self.selection_strategy == 'rank':
           
            indexed_fitnesses = []
            for i in range(population_size):
                indexed_fitnesses.append((fitnesses[i], i))

           
            indexed_fitnesses.sort(key=lambda x: x[0])

            
            
            total_rank = population_size * (population_size + 1) / 2
            
            if total_rank == 0: 
                 return random.randrange(population_size)

            pick = random.uniform(0, total_rank)
            current_rank_sum = 0
            for rank, (_, original_index) in enumerate(indexed_fitnesses, 1): 
                current_rank_sum += rank
                if current_rank_sum >= pick:
                    return original_index
           
            return indexed_fitnesses[-1][1] 
            
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Perform permutation-preserving crossover between two parents.
        Args:
            parent1 (List[int]): First parent individual.
            parent2 (List[int]): Second parent individual.
        Returns:
            List[int]: Offspring individual produced by crossover.
        """
        n = len(parent1)
        offspring = [-1] * n 

        if self.crossover_strategy == 'one_point': 
            cp = random.randint(1, n - 1) 
            
          
            offspring[:cp] = parent1[:cp]
            
           
            fill_idx = cp
            for gene in parent2:
                if gene not in offspring[:cp]:
                    if fill_idx < n:
                        offspring[fill_idx] = gene
                        fill_idx += 1
            return offspring

        elif self.crossover_strategy == 'two_point': 
            cp1, cp2 = sorted(random.sample(range(n), 2))
            
          
            offspring[cp1:cp2+1] = parent1[cp1:cp2+1]
            
           
            genes_from_p1_segment = set(parent1[cp1:cp2+1])
            
           
            current_p2_idx = 0
            for i in range(n):
              
                offspring_idx = (cp2 + 1 + i) % n 
                if offspring[offspring_idx] == -1: 
                    while parent2[current_p2_idx] in genes_from_p1_segment:
                        current_p2_idx += 1
                    offspring[offspring_idx] = parent2[current_p2_idx]
                    current_p2_idx += 1
            return offspring

        elif self.crossover_strategy == 'uniform': 
            positions_from_p1 = sorted(random.sample(range(n), random.randint(1, n -1))) 
            
            genes_from_p1 = {} 
            for pos in positions_from_p1:
                offspring[pos] = parent1[pos]
                genes_from_p1[parent1[pos]] = True

            
            current_p2_idx = 0
            for i in range(n):
                if offspring[i] == -1: 
                    while parent2[current_p2_idx] in genes_from_p1:
                        current_p2_idx += 1
                    offspring[i] = parent2[current_p2_idx]
                    current_p2_idx += 1
            return offspring
            
        else:
            raise ValueError(f"Unknown crossover strategy: {self.crossover_strategy}")


    def mutation(self, individual: List[int]) -> List[int]:
        """
        Apply mutation to an individual using the configured mutation strategy.

        Args:
            individual (List[int]): Individual to mutate.

        Returns:
            List[int]: Mutated individual.
        """
        mutated_individual = individual[:] 
        n = len(mutated_individual)

        if n < 2: 
            return mutated_individual

        if self.mutation_strategy == 'swap':
           
            idx1, idx2 = random.sample(range(n), 2)
           
            mutated_individual[idx1], mutated_individual[idx2] = mutated_individual[idx2], mutated_individual[idx1]
        
        elif self.mutation_strategy == 'inversion':
            
            idx1, idx2 = sorted(random.sample(range(n), 2))
          
            segment_to_invert = mutated_individual[idx1:idx2+1]
            segment_to_invert.reverse()
            mutated_individual[idx1:idx2+1] = segment_to_invert
            
        elif self.mutation_strategy == 'scramble':
            
            idx1, idx2 = sorted(random.sample(range(n), 2))
           
            segment_to_scramble = mutated_individual[idx1:idx2+1]
            random.shuffle(segment_to_scramble)
            mutated_individual[idx1:idx2+1] = segment_to_scramble
            
        else:
            raise ValueError(f"Unknown mutation strategy: {self.mutation_strategy}")
            
        return mutated_individual
    
    def run(self) -> Tuple[List[int], float, List[float], List[float]]:
        """
        Run the genetic algorithm for the configured number of generations.
        """
        if not self.population:
            raise ValueError("Initial population is empty.")

        best_solution_overall: Optional[List[int]] = None
        best_fitness_overall: float = -float('inf')

        self.best_fitness_history = []
        self.best_distance_history = []

        for generation in range(self.num_generations):
        
            fitness_values = self.evaluate_population()

            
            current_gen_best_fitness = -float('inf')
            current_gen_best_individual = None
            
            for i in range(len(self.population)):
                if fitness_values[i] > current_gen_best_fitness:
                    current_gen_best_fitness = fitness_values[i]
                    current_gen_best_individual = self.population[i]

            if current_gen_best_individual is not None: 
                self.best_fitness_history.append(current_gen_best_fitness)
                if current_gen_best_fitness > 0:
                    self.best_distance_history.append(1.0 / current_gen_best_fitness)
                else:
                   
                    self.best_distance_history.append(float('inf')) 

                if current_gen_best_fitness > best_fitness_overall:
                    best_fitness_overall = current_gen_best_fitness
                    best_solution_overall = current_gen_best_individual[:]
            else:
                self.best_fitness_history.append(-float('inf'))
                self.best_distance_history.append(float('inf'))


          
            new_population: List[List[int]] = []
            

            for _ in range(len(self.population)): 
               
                parent1_idx = self.selection(fitness_values)
                parent2_idx = self.selection(fitness_values)
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]

                
                offspring = parent1[:] 
                if random.random() < self.crossover_rate:
                    offspring = self.crossover(parent1, parent2)
                
               
                if random.random() < self.mutation_rate:
                    offspring = self.mutation(offspring)
                
                new_population.append(offspring)

            
            self.population = new_population
            


        if best_solution_overall is None and self.population: 
            
            final_fitnesses = self.evaluate_population()
            best_idx_final = -1
            best_fitness_final = -float('inf')
            for i in range(len(self.population)):
                if final_fitnesses[i] > best_fitness_final:
                    best_fitness_final = final_fitnesses[i]
                    best_idx_final = i
            if best_idx_final != -1:
                best_solution_overall = self.population[best_idx_final][:]
                best_fitness_overall = best_fitness_final


        if best_solution_overall is None: 
            
            return [], 0.0, self.best_fitness_history, self.best_distance_history


        return best_solution_overall, best_fitness_overall, self.best_fitness_history, self.best_distance_history