'''
Evolution module
A Collection of genetic algorithm methods
'''

import math

from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def dominates(fitness_p: list[float],
              fitness_q: list[float],
              directions: list[str]
              ) -> bool:
    '''
    Check if p dominates q where p >= q for every objective
    (depending on direction) and p > q in at least one objective

    Parameters:
        fitness_p: List of fitnesses or objectives of individual p
        fitness_q: List of fitnesses of objectives of individual q
        directions: List of str ("maximize", "minimize"). Whether to
            maximize or minimize for each objective

    Return:
        bool: True if all fitnesses of p better or same as those of q and
            any one of p is better than q otherwise False
    '''

    def compare(fp, fq, direction):
        if direction == 'maximize':
            return fp >= fq
        if direction == 'minimize':
            return fp <= fq

        raise ValueError(f"Unknown direction: {direction}")

    def strictly_better(fp, fq, direction):
        if direction == 'maximize':
            return fp > fq
        if direction == 'minimize':
            return fp < fq

        raise ValueError(f"Unknown direction: {direction}")

    return all(compare(fp, fq, direction) for fp, fq, direction in
               zip(fitness_p, fitness_q, directions)) and \
        any(strictly_better(fp, fq, direction) for fp, fq, direction in
            zip(fitness_p, fitness_q, directions))


def non_dominated_sorting(fitnesses: list[list[float]],
                          directions: list[str]
                          ) -> list[list[int]]:
    """
    Perform non-dominated sorting on a population based on
    their fitness values.

    Parameters:
        fitnesses (list): List of fitness values corresponding to
        each solution in the population.
        directions (list): List of direction for each fitness value
        ('maximize' or 'minimize').

    Returns:
        list: A list of fronts where each front contains indices of
        non-dominated solutions.
    """
    fronts = [[]]
    domination_count = [0] * len(fitnesses)
    dominated_solutions = [[] for _ in range(len(fitnesses))]

    for p, fitness_p in enumerate(fitnesses):
        for q, fitness_q in enumerate(fitnesses):
            if dominates(fitness_p, fitness_q, directions):
                dominated_solutions[p].append(q)
            elif dominates(fitness_p=fitness_q,
                           fitness_q=fitness_p,
                           directions=directions):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()  # Remove the last empty front
    return fronts


def crowding_distance(front: list[int],
                      fitnesses: list[list[float]]
                      ) -> list[float]:
    """
        Calculate the crowding distance for each solution in a Pareto front
        based on the fitness values of the objectives.

        Parameters:
        - front: A list of indices representing the solutions in
        the Pareto front.
        - fitnesses: A list of lists where each inner list contains
        the fitness values of each solution for all objectives.

        Returns:
        - A list of crowding distances for each solution in the front.
    """
    # There is only one individual in the front, set the distance to infinity
    if len(front) == 1:
        return [float('inf')]

    # Initialize a dictionary to store the crowding distance for
    # each individual in the front
    distance_dict = {i: 0 for i in front}
    num_objectives = len(fitnesses[0])

    for m in range(num_objectives):
        sorted_front = sorted(front, key=lambda i, m=m: fitnesses[i][m])
        min_fitness, max_fitness = \
            fitnesses[sorted_front[0]][m], fitnesses[sorted_front[-1]][m]
        # Set crowding distance to infinity for boundary solutions
        distance_dict[sorted_front[0]] = float('inf')
        distance_dict[sorted_front[-1]] = float('inf')

        for i in range(1, len(front) - 1):
            if max_fitness != min_fitness:
                distance_dict[sorted_front[i]] += \
                    (fitnesses[sorted_front[i + 1]][m] -
                     fitnesses[sorted_front[i - 1]][m]) / \
                    (max_fitness - min_fitness)

    # Map the crowding distances back to a list in the original front order
    return [distance_dict[i] for i in front]


def reference_point_generation(num_objectives: int,
                               divisions: int
                               ) -> np.ndarray:
    """
    Generate reference points uniformly distributed in the objective space.
    """
    points = []

    def recursive_generate(n, left_sum, current_point):
        if n == 1:
            points.append(current_point + [left_sum / divisions])
        else:
            for i in range(left_sum + 1):
                recursive_generate(n - 1, left_sum - i,
                                   current_point + [i / divisions])

    recursive_generate(num_objectives, divisions, [])
    return np.array(points)


def calculate_num_reference_points(num_objectives: int, divisions: int) -> int:
    """
    Calculate the number of reference points for NSGA-III based on
    the number of objectives (m) and the number of divisions (n).

    Parameters:
    - num_objectives (int): Number of objectives.
    - divisions(int): Number of divisions.

    Returns:
    - int: Total number of reference points.
    """
    return math.comb(num_objectives + divisions - 1, divisions)


def associate_to_reference_points(fitnesses: list[list[float]],
                                  reference_points: np.ndarray
                                  ) -> list[int]:
    """
    Associate individuals to reference points based on
    minimum Euclidean distance.
    Return a list of reference points that individuals are associated with
    For example, if associations = [3, 7, 9] -> first individual is closed
    to reference point 3, second one is near point 7, and last one is in
    range of point 9
    """
    associations = []
    for fitness in fitnesses:
        distances = np.linalg.norm(reference_points - fitness, axis=1)
        closest_point = np.argmin(distances)
        associations.append(closest_point)
    return associations


def nsga2_selection(front: list[int],
                    crowd_distances: list[float],
                    n_selections: int
                    ) -> list[int]:
    """
        Select individuals in the Pareto front using
        crowd distances (NSGA-II approach)
    """
    # Sort individuals in the current front based on crowding distance
    # (larger distance means less crowding)
    sorted_by_distance = sorted(range(len(front)),
                                key=lambda i: crowd_distances[i],
                                reverse=True)
    # Select the least crowded individuals from the current front until
    # num_parents is reached
    return [front[i] for i in sorted_by_distance[:n_selections]]


def nsga3_selection(front: list[int],
                    associations: list[int],
                    n_selections: int
                    ) -> list[int]:
    """
    Select individuals based on their association with reference points to
    maintain diversity, prioritizing individuals associated with the least
    populated reference points.

    Parameters:
        front (list[int]): List of indices representing individuals in the
                           current front (subset of the population).
        associations (list[int]): List where each element corresponds to a
                                  reference point association for each
                                  individual in `front`.
        n_selections (int): Number of individuals to select from the `front`.

    Returns:
        list[int]: List of selected individual indices from `front`, chosen to
                   maximize diversity based on reference point associations.
    """
    # Count individuals associated with each reference point
    reference_count = {assoc: 0 for assoc in set(associations)}
    for assoc in associations:
        reference_count[assoc] += 1

    # Sort front indices by the association count of their reference point
    sorted_front = sorted(
        range(len(front)),
        key=lambda i: reference_count[associations[i]]
    )

    # Select the required number of individuals by slicing the sorted list
    selected_indices = [front[i] for i in sorted_front[:n_selections]]
    return selected_indices


def tournament_selection_nsga2(selected_indices: list[int],
                               fronts: list[list[int]],
                               crowd_distances_all: list[list[float]],
                               num_to_select: int,
                               tournament_size: int,
                               tournament_replace: bool) -> list[int]:
    """
        Perform tournament selection based on Pareto rank and
        crowding distance.

        Parameters:
            selected_indices (list[int]): A list of selected indices
                from NSGA-II.
            fronts (list[list[int]]): The Pareto fronts of the population.
            fitnesses (list[list[float]]): The fitness values for
                each objective for the population.
            num_to_select (int): The number of individuals to select for
                the next generation.
            tournament_size (int): The number of individuals to participate in
                each tournament round.
            replace (bool): If True, individuals can be selected multiple
                times. If False, individuals are removed from the pool
                once selected.

        Returns:
            list[int]: A list of selected indices after tournament selection.
    """

    selected = []
    while len(selected) < num_to_select:
        # Tournament size cannot be larger than number of input population
        tournament_size = min(tournament_size, len(selected_indices))
        # Randomly sample individuals for the tournament
        tournament_contenders = np.random.choice(selected_indices,
                                                 tournament_size,
                                                 replace=False)

        # Initialize variables to track the best individual in
        # the tournament
        best_individual = tournament_contenders[0]
        best_front = next(i for i, front in enumerate(fronts)
                          if best_individual in front)

        # Iterate through the tournament contenders
        for contender in tournament_contenders[1:]:
            contender_front = next(i for i, front in enumerate(fronts)
                                   if contender in front)

            # Compare Pareto rank (lower front number is better)
            if contender_front < best_front:
                best_individual = contender
                best_front = contender_front
            elif contender_front == best_front:
                # If they are in the same front, use crowding distance
                # as a tie-breaker
                contender_distance = crowd_distances_all[best_front][
                    fronts[best_front].index(contender)]
                best_distance = crowd_distances_all[best_front][
                    fronts[best_front].index(best_individual)]

                if contender_distance > best_distance:
                    best_individual = contender

        # Add the best individual to the selected list
        selected.append(best_individual)

        # Remove the selected individual from the pool if replace is False
        if not tournament_replace:
            selected_indices.remove(best_individual)

    return selected


def tournament_selection_nsga3(selected_indices: list[int],
                               fronts: list[list[int]],
                               associations_all: list[list[int]],
                               num_to_select: int,
                               tournament_size: int,
                               tournament_replace: bool) -> list[int]:
    """
    Perform tournament selection based on NSGA-III reference point association.

    Parameters:
        selected_indices (list[int]): A list of selected indices.
        fronts (list[list[int]]): The Pareto fronts of the population.
        associations (list[list[int]]): A list of associations of each front
        num_to_select (int): Number of individuals to select for the next
        generation.
        tournament_size (int): Number of individuals in each tournament.
        tournament_replace (bool): If True, individuals can be selected
        multiple times; otherwise, they are removed.

    Returns:
        list[int]: List of selected indices after tournament selection.
    """
    selected = []
    while len(selected) < num_to_select:
        # Tournament size cannot exceed available candidates
        tournament_size = min(tournament_size, len(selected_indices))

        # Randomly sample individuals for the tournament
        tournament_contenders = np.random.choice(selected_indices,
                                                 tournament_size,
                                                 replace=False)
        best_individual = tournament_contenders[0]
        best_front = next(i for i, front in enumerate(fronts)
                          if best_individual in front)

        for contender in tournament_contenders[1:]:
            contender_front = next(i for i, front in enumerate(fronts)
                                   if contender in front)

            # Select based on Pareto rank
            if contender_front < best_front:
                best_individual = contender
                best_front = contender_front
            elif contender_front == best_front:
                # If they are in the same front, prefer less crowded
                # reference points
                associations = associations_all[best_front]
                contender_association = associations[
                    fronts[best_front].index(contender)]
                best_association = associations[
                    fronts[best_front].index(best_individual)]
                if associations.count(contender_association) < \
                        associations.count(best_association):
                    best_individual = contender

        # Add the best individual to the selected list
        selected.append(best_individual)

        # Remove selected individual if replacement is not allowed
        if not tournament_replace:
            selected_indices.remove(best_individual)

    return selected


class GeneticAlgorithmMixin(ABC):
    '''
    Collection of genetic algorithm methods that will be implemented
    in prescriptors
    '''
    def selection(self,
                  method: str,
                  population: list,
                  fitnesses: list[list[float]],
                  directions: list[str],
                  selection_size: int | float | str,
                  divisions: int,
                  tournament_size: int | float,
                  tournament_replace: bool
                  ) -> list:
        """
        Perform selection to choose a set of parents for the next generation
        based on selected method.

        Parameters:
            method (str): Available methods are 'nsga2' and 'nsga3'
            population (list): A list of individuals (neural network weights)
                represented as tensors.
            fitnesses (list[list[float]]): A list of lists,
                where each sublist contains the fitness values for
                each objective for the corresponding individual
                in the population.
            directions (list): A list of directions to optimize
                ('maximize' or 'minimize').
            selection_size: Number of parents to select. If:
                - int: Select exact number of parents.
                - float: Select proportion of the population
                    (e.g., 0.5 = 50% of population).
                - str: Select up to a specific Pareto front.
                    '0' = select first front, '1' = first and second fronts.
            divisions (int): Number of division per objective for
                nsga3 method
            tournament_size (int or float): If int, number of individual in
                tournament. if float, ratio of population in tournament
            tournament_replace (bool): If True, individuals can be selected
                multiple times. If False, individuals are removed from
                the pool once selected.

        Returns:
            list: A list of selected parents (individuals) from the population.
        """

        # Handle different types of num_parents
        if isinstance(selection_size, float):
            # If num_parents is a float, select that proportion of
            # the population
            selection_size = max(1, int(len(population) * selection_size))
            front_limit = None
        elif isinstance(selection_size, str):
            # If num_parents is a string, interpret it as the number of fronts
            # to select from
            try:
                front_limit = int(selection_size)
                selection_size = len(population)
            except ValueError as exc:
                raise ValueError(
                    "num_parents string must be a valid integer "
                    "representing the front number."
                ) from exc
        else:
            # If num_parents is an integer, leave it as is
            front_limit = None
        # Perform non-dominated sorting to get Pareto fronts
        fronts = non_dominated_sorting(fitnesses, directions)
        crowd_distances_all = None
        associations_all = None
        if method in \
                ['nsga2', 'nsga3', 'tournament_nsga2', 'tournament_nsga3']:
            if method in ['nsga2', 'tournament_nsga2']:
                # Compute crowding distances for all fronts
                crowd_distances_all = [crowding_distance(front, fitnesses)
                                       for front in fronts]
            else:
                # Generate reference points
                reference_points = reference_point_generation(
                    len(fitnesses[0]), divisions)
                # Scale fitnesses from 0 to 1
                scaled_fitnesses = \
                    MinMaxScaler().fit_transform(np.array(fitnesses))
                # Compute associations for all fronts
                associations_all = [associate_to_reference_points(
                    [scaled_fitnesses[i] for i in front], reference_points)
                    for front in fronts]
        else:
            raise ValueError(f"{method} is not in "
                             "['nsga2', 'nsga3', "
                             "'touranment_nsga2', 'tournament_nsga3']")

        selected = []
        # Iterate over each front and select individuals
        for i, front in enumerate(fronts):
            if front_limit is not None and i > front_limit:
                break  # Stop once we reach the specified front limit
            if len(front) + len(selected) > selection_size:
                remaining_slots = selection_size - len(selected)
                if method in ['nsga3', 'tournament_nsga3']:
                    selected.extend(
                        nsga3_selection(front,
                                        associations_all[i],
                                        remaining_slots)
                                        )
                else:
                    selected.extend(
                        nsga2_selection(front,
                                        crowd_distances_all[i],
                                        remaining_slots)
                                        )
                break
            # Add the entire front to the selected individuals
            # if space allows
            selected.extend(front)
            if len(selected) == selection_size:
                break

        if method.startswith('tournament'):
            if isinstance(tournament_size, float):
                tournament_size = max(2,
                                      int(len(population) * tournament_size))
            else:
                max(2, tournament_size)
            if method == 'tournament_nsga2':
                selected = tournament_selection_nsga2(selected,
                                                      fronts,
                                                      crowd_distances_all,
                                                      len(selected),
                                                      tournament_size,
                                                      tournament_replace)
            else:
                selected = tournament_selection_nsga3(selected,
                                                      fronts,
                                                      associations_all,
                                                      len(selected),
                                                      tournament_size,
                                                      tournament_replace)

        # Return the selected individuals from the population
        return [population[i] for i in selected]

    def evolve(self,
               directions: dict,
               method: str,
               selection_size,
               population_size: int,
               n_generations: int,
               elite_ratio: float,
               crossover_method: str,
               mutation_rate: float,
               mutation_replace: list[bool],
               mutation_specs: list[dict],
               divisions: int,
               tournament_size: int | float,
               tournament_replace: bool,
               max_attempts: int,
               verbosity: int
               ) -> tuple[list, list[list[float]]]:
        '''Evolve population through generations'''

        directions_list = list(directions.values())
        terminate_search = False
        seen_individuals = set()
        # To prevent initial population from having duplicates
        attempts = 0
        while len(seen_individuals) != population_size:
            population = self.initialize_population(population_size)
            seen_individuals = set(tuple(individual)
                                   for individual in population)
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(f"Cannot initialize {population_size} "
                                 "non-duplicated individuals "
                                 f"after maximum attempts ({max_attempts})")
        n_elites = int(elite_ratio*population_size)
        # For elites (no crossover or mutation), tournament is the same as
        # nsga2 but more costly
        if method.startswith('tournament'):
            elites_method = method.split('_')[1]
        else:
            elites_method = method
        # Check if the fitness_function works on entire population
        # or individuals
        # Call the helper method to get fitnesses
        fitnesses = self.fitness_function(population)

        for generation in range(n_generations):
            # To prevent duplicated elites in case method == 'tournament',
            # set tournament_replace to False
            elites = self.selection(elites_method,
                                    population,
                                    fitnesses,
                                    directions_list,
                                    n_elites,
                                    divisions,
                                    tournament_size,
                                    tournament_replace=False)

            # Default behavior when num_parents is not specified
            selection_size = len(population) - n_elites \
                if selection_size is None else selection_size

            parents = self.selection(method,
                                     population,
                                     fitnesses,
                                     directions_list,
                                     selection_size,
                                     divisions,
                                     tournament_size,
                                     tournament_replace)

            # If len(parents) is odd, duplicate the first parent
            if len(parents) % 2 != 0:
                parents.append(parents[0])

            # Reproduction
            new_population = []
            attempts = 0
            while len(new_population) < population_size-len(elites) and \
                    not terminate_search:
                for i in range(0, len(parents), 2):
                    parent1, parent2 = parents[i], parents[i+1]
                    for child in self.crossover(parent1,
                                                parent2,
                                                crossover_method):
                        mutated_child = self.mutate(child,
                                                    mutation_rate,
                                                    mutation_replace,
                                                    mutation_specs)
                        if tuple(mutated_child) not in seen_individuals:
                            new_population.append(mutated_child)
                            seen_individuals.add(tuple(mutated_child))
                            # Reset attempts when a new individual is found
                            attempts = 0
                        else:
                            attempts += 1

                        # If we've reached the max number of attempts,
                        # stop to avoid infinite loop
                        if attempts > max_attempts:
                            if verbosity != -1:
                                print("Warning: Search terminated prematurely "
                                      f"in generation {generation+1} "
                                      "due to inability to generate new unique"
                                      " solutions after maximum attempts"
                                      f"({max_attempts}).")
                            terminate_search = True
                            break

                    if terminate_search:
                        break
                # Shuffle parents in case more population is needed
                np.random.shuffle(parents)

            if terminate_search:
                break

            # Ensure population size is maintained
            population = (elites + new_population)[:population_size]
            # Calculate fitness
            fitnesses = self.fitness_function(population)

            if verbosity == 1:
                # Report progress
                print(f'Generation {generation+1}')
                for i, outcome_direction in enumerate(directions.items()):
                    best_fitness = None
                    if outcome_direction[1] == 'maximize':
                        best_fitness = max(np.array(fitnesses).T[i])
                    else:
                        best_fitness = min(np.array(fitnesses).T[i])
                    avg_fitness = np.mean(np.array(fitnesses).T[i])
                    print(f'Best {outcome_direction[0]}: {best_fitness}, '
                          f'Average {outcome_direction[0]}: {avg_fitness}')

        return population, fitnesses

    def crossover(self,
                  parent1: list,
                  parent2: list,
                  method: str
                  ) -> tuple[list]:
        """
        Apply crossover between two parent individuals based on
        the chosen method.

        Parameters:
        - method (str): The crossover method to use
            ('onepoint', 'twopoint', 'uniform').
        - parent1 (list): The first parent individual.
        - parent2 (list): The second parent individual.

        Returns:
        - tuple: A tuple containing two offspring individuals (child1, child2).
        """
        length = len(parent1)

        if method == 'onepoint':
            # One-point crossover
            point = np.random.randint(1, length - 1)
            return (parent1[:point] + parent2[point:],
                    parent2[:point] + parent1[point:])

        if method == 'twopoint':
            # Two-point crossover
            point1, point2 = sorted(np.random.randint(1, length - 1, size=2))
            return (parent1[:point1] +
                    parent2[point1:point2] +
                    parent1[point2:],
                    parent2[:point1] +
                    parent1[point1:point2] +
                    parent2[point2:])

        if method == 'uniform':
            # Uniform crossover
            child1, child2 = zip(*[(p1, p2) if np.random.rand() < 0.5
                                   else (p2, p1) for p1, p2 in
                                   zip(parent1, parent2)])
            return list(child1), list(child2)

        raise ValueError(f"Unknown crossover method: {method}")

    def mutate(self,
               individual,
               mutation_rate: float,
               mutation_replace: list[bool],
               mutation_specs: list[dict]
               ) -> list:
        """
        Apply mutation to an individual with mixed gene types based on
        specific mutation parameters.

        Parameters:
        - individual (list): The individual to be mutated, potentially
            containing mixed data types.
        - mutation_rate (float): The probability of mutating each gene.
        - replace (bool): If True, replaces the gene with a new value.
        If False, adds noise to numeric genes.
        - mutation_specs (list[dict]): A list of dictionaries, each specifying
            mutation details for a corresponding gene:
            - 'method' (str): Mutation method ('gaussian', 'uniform', 'choice')
            - 'param' (tuple): Parameters for the mutation:
                - For 'gaussian': (mean, standard deviation).
                - For 'uniform': (lower bound, upper bound).
                - For 'choice': A list of possible replacement values
                    for categorical genes.

        Returns:
        - list: The mutated individual.
        """
        mutated_individual = []

        for gene, replace, spec in zip(individual,
                                       mutation_replace,
                                       mutation_specs):
            if np.random.rand() < mutation_rate:
                method, param = spec['method'], spec['param']

                if method == 'gaussian' and isinstance(gene, (int, float)):
                    if replace:
                        new_value = param[0] + param[1] * np.random.randn()
                    else:
                        new_value = gene + \
                            param[1]*np.random.randn() + param[0]

                elif method == 'uniform' and isinstance(gene, (int, float)):
                    if replace:
                        new_value = np.random.uniform(param[0], param[1])
                    else:
                        new_value = gene + \
                            np.random.uniform(param[0], param[1])

                elif method == 'choice':
                    if replace:
                        new_value = np.random.choice(param)
                    else:
                        new_value = gene

                else:
                    raise ValueError(f"""Invalid mutation method '{method}'
                                     for gene '{gene}'
                                     with type {type(gene)}.""")

                mutated_individual.append(new_value)
            else:
                mutated_individual.append(gene)

        return mutated_individual

    def initialize_population(self, population_size: int) -> list:
        """Initialize a population with a given population size."""
        population = []
        for _ in range(population_size):
            individual = self.generate_individual()
            population.append(individual)

        return population

    @abstractmethod
    def generate_individual(self):
        """Abstract method for generating an individual.
        Must be implemented by child classes."""
        return

    @abstractmethod
    def fitness_function(self, population):
        """Abstract method for calculate fitness of population.
        Must be implemented by child classes."""
        return
