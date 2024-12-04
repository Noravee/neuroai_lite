'''Module containing prescriptor classes'''
from itertools import product

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler,
                                   QuantileTransformer,
                                   TargetEncoder
                                   )
from sklearn.random_projection import GaussianRandomProjection
import torch
from torch import nn

from encoder import DataEncoder
from evolution import GeneticAlgorithmMixin, non_dominated_sorting
from neuralnet import ForwardNeuralNet
from customsklearn import CustomTargetEncoder

DETAILED = 1
SUMMARY = 0
SILENCE = -1


class NeuralNetworkPrescriptor(GeneticAlgorithmMixin):
    '''Prescriptor based on neural network'''
    def __init__(
            self,
            predictors: dict = None,
            cao_dict: dict = None,
            directions: dict = None,
            method: str = 'tournament_nsga2',
            selection_size: int | float | str = None,
            n_hidden_layer: int = 1,
            evolve_activations: bool = False,
            population_size: int = 100,
            n_generations: int = 100,
            elite_ratio: float = 0.1,
            crossover_method: str = 'uniform',
            mutation_method: str = 'gaussian',
            mutation_rate: float = 0.1,
            mutation_replace: bool = False,
            mutation_params: tuple = (0, 0.1),
            divisions: int = 5,
            tournament_size: float | int = 2,
            tournament_replace: bool = True,
            max_attempts: int = 100,
            use_gpu: bool = False,
            verbosity: int = DETAILED,
            random_state: int = None
                ) -> None:
        """
        Initialize the TournamentPrescriptor class.

        Parameters:
            predictors: The predictor for making predictions from
                context and actions.
            use_gpu: A boolean indicating whether to use GPU
                (default is False).
            random_state: An integer for setting random seed (default is None).

        Returns:
            None
        """

        if cao_dict is not None:
            if set(cao_dict.keys()) != {'context', 'actions', 'outcomes'}:
                raise ValueError(
                    "Keys of cao_dict has to be "
                    "'context', 'actions', and 'outcomes'"
                )
            for k, v in cao_dict.items():
                if not isinstance(v, list):
                    raise TypeError(f"Value for '{k}' must be a list.")
                if not all(isinstance(item, str) for item in v):
                    raise TypeError(
                        f"All elements in the '{k}' list must be strings.")
        self.cao_dict = cao_dict

        if predictors is not None and cao_dict is not None:
            if not isinstance(predictors, dict):
                raise TypeError(
                    "'Predictors' has to be a dictionary with outcomes as "
                    "keys and Predictor as values"
                )
            if set(predictors.keys()) != set(cao_dict['outcomes']):
                raise ValueError(
                    "Keys in predictors and values of outcomes of cao_dict "
                    "have to be the same"
                )
            # Reorder predictors according to cao_dict['outcomes']
            predictors = {outcome: predictors[outcome]
                          for outcome in cao_dict['outcomes']}
        self.predictors = predictors

        if directions is not None and cao_dict is not None:
            if not isinstance(directions, dict):
                raise TypeError(
                    "'directions' has to be a dictionary with outcomes as keys"
                    " and 'maximize' or 'minimize' as values"
                )
            if set(directions.keys()) != set(cao_dict['outcomes']):
                raise ValueError(
                    "Keys in directions and values of outcomes of "
                    "cao_dict have to be the same"
                )
            for k, v in directions.items():
                if v not in ['maximize', 'minimize']:
                    raise ValueError(
                        f"Direction of {k} is neither "
                        "'maximize' nor 'minimize'"
                    )
            # Reorder directions according to cao_dict['outcomes']
            directions = {outcome: directions[outcome]
                          for outcome in cao_dict['outcomes']}
        self.directions = directions
        self.method = method
        self.selection_size = selection_size
        self.n_hidden_layer = n_hidden_layer
        self.evolve_activations = evolve_activations
        self.population_size = population_size
        self.n_generations = n_generations
        self.elite_ratio = elite_ratio
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.mutation_rate = mutation_rate
        self.mutation_replace = mutation_replace
        self.mutation_params = mutation_params
        self.divisions = divisions
        self.tournament_size = tournament_size
        self.tournament_replace = tournament_replace
        self.max_attempts = max_attempts
        self.verbosity = verbosity
        self.random_state = random_state

        self.context_df = None
        self.context_encoder = None
        self.actions_encoder = None
        self.model = None
        self.best_individuals = None
        self.pareto_df = None

        if use_gpu:
            # Check for CUDA
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            # Check for MPS (Metal Performance Shaders) for Apple devices
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            # Default to CPU
            else:
                self.device = torch.device("cpu")
                if verbosity != SILENCE:
                    print('No GPU device is detected.')
        else:
            self.device = torch.device('cpu')
        if verbosity != SILENCE:
            # Print the selected device
            print(f"Device set to: {self.device}")

    def generate_individual(self):
        individual = self.n_hidden_layer*['Tanh']
        for layer in self.model.network:
            if isinstance(layer, nn.Linear):
                if len(self.cao_dict['context']) == 1 and \
                        len(self.cao_dict['actions']) == 1:
                    nn.init.normal_(layer.weight)
                else:
                    nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
                for param in layer.parameters():
                    individual.extend(param.data.flatten().tolist())

        return individual

    def update_weights_and_activations(self,
                                       individual: list[nn.Module, float]):
        '''
        Update the weights and activations of the neural network model
        based on the provided individual.

        Parameters:
            individual (list[nn.Module, float]): List containing activation
            functions and weights for each layer.

        Raises:
            ValueError: If there is a mismatch between the provided weights
            and the model's parameters or the number of new activations
            and the model's activations.
        '''
        new_activation_names = individual[:self.n_hidden_layer]
        new_activations = []
        for activation_name in new_activation_names:
            if activation_name == 'Linear':
                activation = None  # No activation function for Linear
            else:
                activation = getattr(nn, activation_name)()
            new_activations.append(activation)
        new_weights = torch.tensor(individual[self.n_hidden_layer:],
                                   dtype=torch.float32)
        offset = 0
        act_idx = 0
        for i, layer in enumerate(self.model.network):
            if isinstance(layer, nn.Linear):
                # Update weights and biases for the layer
                weight_shape = layer.weight.shape
                bias_shape = layer.bias.shape

                layer.weight.data = \
                    new_weights[offset:offset + layer.weight.numel()]\
                    .reshape(weight_shape).clone().detach()
                offset += layer.weight.numel()

                layer.bias.data = \
                    new_weights[offset:offset + layer.bias.numel()]\
                    .reshape(bias_shape).clone().detach()
                offset += layer.bias.numel()

            elif isinstance(layer, nn.Module) and\
                not isinstance(layer, nn.Linear) and\
                    act_idx < len(new_activations):
                # Only update activation if the new activation is not None
                if act_idx < len(new_activations):
                    new_activation = new_activations[act_idx]
                    if new_activation is not None:
                        self.model.network[i] = new_activation
                    act_idx += 1

        if offset != new_weights.numel():
            raise ValueError(
                "Mismatch between the provided weights and "
                "the model's parameters."
            )
        if act_idx != len(new_activations):
            raise ValueError(
                "Mismatch between the number of new activations and "
                "the model's activations."
            )

    def prescribe(
            self,
            context_df: pd.DataFrame,
            prescriptor_id: int
            ) -> pd.DataFrame:
        """
        Prescribes actions based on the given context data and
        predicts outcomes for a specific prescriptor.

        Parameters:
            context_df (pd.DataFrame): The dataframe containing context data.
            prescriptor_id (int): The ID of the prescriptor to use for
            prescribing actions.

        Returns:
            pd.DataFrame: A dataframe containing the prescribed actions for
            the given context.
        """

        # Check if 'best_individuals' exists
        if not hasattr(self, 'best_individuals') or\
                self.best_individuals is None:
            raise ValueError(
                "Error: 'best_individuals' does not exist. "
                "Train the model using the 'fit' method or "
                "load models using the 'load' method."
            )

        # Check if 'prescriptor_id' is within range
        if not 0 <= prescriptor_id < len(self.best_individuals):
            raise IndexError(
                "Error: Invalid 'prescriptor_id'. Available indices are "
                f"from 0 to {len(self.best_individuals) - 1}."
            )

        # Assign weights and prescribe actions
        self.update_weights_and_activations(
            self.best_individuals[prescriptor_id])
        actions_df = self.__prescribe(context_df)
        actions_df.insert(0, 'ID', [prescriptor_id]*len(context_df))

        return actions_df

    def prescribe_predict(self,
                          context_df: pd.DataFrame,
                          prescriptor_id: int
                          ) -> pd.DataFrame:
        """
        Prescribes actions based on the given context data and
        predicts outcomes.

        Parameters:
            context_df (pd.DataFrame): The dataframe containing context data.
            prescriptor_id (int): The ID of the prescriptor to use for
            prescribing actions.

        Returns:
            pd.DataFrame: A dataframe containing prescribed actions and
            predicted outcomes.
        """
        actions_df = self.prescribe(context_df, prescriptor_id)
        ca_df = pd.concat([context_df, actions_df], axis=1)
        outcomes_df = self.__predict(ca_df)
        ao_df = pd.concat([actions_df, outcomes_df], axis=1)
        return ao_df

    def prescribe_predict_all(self, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prescribes actions and predicts outcomes for all prescriptors in
        the Pareto front.

        Parameters:
            context_df (pd.DataFrame): The dataframe containing context data.

        Returns:
            pd.DataFrame: A dataframe containing prescribed actions and
            predicted outcomes for all prescriptors.
        """
        ao_dfs = []
        for i in self.pareto_df.index:
            ao_df = self.prescribe_predict(context_df, i)
            ao_dfs.append(ao_df)

        return pd.concat(ao_dfs, axis=0).reset_index(drop=True)

    def fitness_function(self, population: list[list]) -> list[list[float]]:
        fitnesses = []
        for individual in population:
            self.update_weights_and_activations(individual)
            cao_df = self.__prescribe_predict(self.context_df)

            outcomes = [cao_df[outcome].mean()
                        for outcome in self.cao_dict['outcomes']]
            fitnesses.append(outcomes)

        return fitnesses

    def _pareto(self,
                fronts: list[list[int]],
                fitnesses: list[list[float]]
                ) -> pd.DataFrame:

        # Extract pareto-optimal fitnesses
        pareto = [fitnesses[i] for i in fronts[0]]

        # Create DataFrame with appropriate outcome columns
        pareto_df = pd.DataFrame(pareto, columns=self.cao_dict['outcomes'])

        # Name the index 'ID'
        pareto_df.index.name = 'ID'

        # Order the DataFrame by the first outcome column
        first_outcome_col = self.cao_dict['outcomes'][0]
        pareto_df = pareto_df.sort_values(by=first_outcome_col)

        return pareto_df.reset_index(drop=False)

    def fit(self, df: pd.DataFrame):
        '''
        Fit the model to the provided DataFrame by creating encoders for
        context and actions data, initializing the neural network model
        with specified layers and activations, evolving the model through
        generations using a genetic algorithm, and identifying the best
        individuals based on non-dominated sorting.
        Print the resulting DataFrame if verbosity is not set to -1.
        '''
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        context = df[self.cao_dict['context']]
        self.context_df = context
        actions = df[self.cao_dict['actions']]
        outcomes = df[self.cao_dict['outcomes']]
        context_ncols = context.shape[1]
        actions_ncols = actions.shape[1]
        if context_ncols >= actions_ncols:
            layers = [context_ncols]*(self.n_hidden_layer+1)+[actions_ncols]
        else:
            layers = [actions_ncols]*(self.n_hidden_layer+2)

        context_encoder, actions_encoder = \
            self.__create_encoders(context, actions, outcomes)
        self.context_encoder = context_encoder
        self.actions_encoder = actions_encoder

        model = ForwardNeuralNet(
            layers=layers,
            activation=nn.Tanh(),
            output_activation=nn.Sigmoid()
            )
        self.model = model

        activation_names = ['Linear', 'Tanh', 'Sigmoid', 'ReLU', 'ELU', 'SELU']

        n_params = sum(p.numel() for p in model.parameters())
        mutation_replace = [self.evolve_activations] + \
            n_params*[self.mutation_replace]
        mutation_specs = \
            [{'method': 'choice', 'param': activation_names}] + \
            n_params*[{'method': self.mutation_method,
                       'param': self.mutation_params}]

        population, fitnesses = self.evolve(self.directions,
                                            self.method,
                                            self.selection_size,
                                            self.population_size,
                                            self.n_generations,
                                            self.elite_ratio,
                                            self.crossover_method,
                                            self.mutation_rate,
                                            mutation_replace,
                                            mutation_specs,
                                            self.divisions,
                                            self.tournament_size,
                                            self.tournament_replace,
                                            self.max_attempts,
                                            self.verbosity,
                                            )
        directions_list = list(self.directions.values())
        fronts = non_dominated_sorting(fitnesses, directions_list)
        self.best_individuals = [population[i] for i in fronts[0]]
        self.pareto_df = self._pareto(fronts, fitnesses)
        if self.verbosity != SILENCE:
            print(self.pareto_df)

    def save(self, file_path: str):
        """
        Save the entire prescriptor (model, encoders, etc.)
        """
        # Save parts of the prescriptor (encoder, predictor, etc.)
        joblib.dump({
            'context_encoder': self.context_encoder,
            'actions_encoder': self.actions_encoder,
            'predictors': self.predictors,
            'best_individuals': self.best_individuals,
            'layers': self.model.layers
        }, file_path + '_components.joblib')

        # Save pareto_df
        self.pareto_df.to_csv(file_path+'_pareto.csv')

    def load(self, file_path: str):
        """
        Load the prescriptor class, model weights, and external components
        """
        # Load parts of the prescriptor
        components = joblib.load(file_path + '_components.joblib')
        self.context_encoder = components['context_encoder']
        self.actions_encoder = components['actions_encoder']
        self.predictors = components['predictors']
        self.best_individuals = components['best_individuals']
        layers = components['layers']
        model = ForwardNeuralNet(
            layers=layers,
            activation=nn.Tanh(),
            output_activation=nn.Sigmoid()
            )
        self.model = model

        # Load pareto_df
        self.pareto_df = pd.read_csv(file_path+'_pareto.csv')

    def __prescribe(self, context_df: pd.DataFrame) -> pd.DataFrame:

        # Transform context
        context_transformed = self.context_encoder.transform(context_df)

        # Prescibe the action
        self.model.to(self.device)
        prescribed_action = \
            self.model\
            .forward(
                torch.from_numpy(context_transformed).float().to(self.device))\
            .to('cpu').numpy()
        self.model.to('cpu')
        actions_df = self.actions_encoder.inverse_transform(prescribed_action)

        return actions_df

    def __predict(self, x: pd.DataFrame) -> pd.DataFrame:
        y_pred = {}
        for outcome, predictor in self.predictors.items():
            y_pred[outcome] = predictor.predict(x)
        return pd.DataFrame(y_pred)

    def __prescribe_predict(self, context_df: pd.DataFrame) -> pd.DataFrame:
        actions_df = self.__prescribe(context_df)
        ca_df = pd.concat([context_df, actions_df], axis=1)
        outcomes_df = self.__predict(ca_df)
        ao_df = pd.concat([actions_df, outcomes_df], axis=1)
        return ao_df

    def __create_encoders(self,
                          context: pd.DataFrame,
                          actions: pd.DataFrame,
                          outcomes: pd.DataFrame
                          ) -> tuple:
        """
        Create encoders for context and actions data based on
        the specified dataframes.

        Parameters:
            context (pd.DataFrame): The dataframe containing context data.
            actions (pd.DataFrame): The dataframe containing actions data.
            outcomes (pd.DataFrame): The dataframe containing outcomes data.

        Returns:
            tuple[Pipeline, DataEncoder]: A tuple containing the Pipeline for
            context data and the DataEncoder for actions data.
        """
        context_ncols = context.shape[1]
        actions_ncols = actions.shape[1]
        context_obj_trans = TargetEncoder()
        context_bool_trans = TargetEncoder()
        actions_num_trans = MinMaxScaler()
        actions_obj_trans = CustomTargetEncoder(target_type='continuous')
        actions_bool_trans = CustomTargetEncoder(target_type='continuous')
        context_trans = DataEncoder(obj_transformer=context_obj_trans,
                                    bool_transformer=context_bool_trans)
        qt = QuantileTransformer(output_distribution='normal', subsample=None)
        pca = PCA(whiten=True, svd_solver='full')
        grp = GaussianRandomProjection(n_components=actions_ncols)
        if context_ncols >= actions_ncols:
            context_steps = [('datatransformers', context_trans),
                             ('qt', qt),
                             ('pca', pca)]
            context_encoder = Pipeline(steps=context_steps)
        else:
            context_steps = [('datatransformers', context_trans),
                             ('grp', grp),
                             ('qt', qt),
                             ('pca', pca)]
            context_encoder = Pipeline(steps=context_steps)
        context_encoder.fit(context)
        actions_encoder = DataEncoder(num_transformer=actions_num_trans,
                                      obj_transformer=actions_obj_trans,
                                      bool_transformer=actions_bool_trans)
        actions_encoder.fit(actions, outcomes)

        return context_encoder, actions_encoder


class SingleSampleContextPrescriptor(GeneticAlgorithmMixin):
    '''SingleContextPrescriptor for optimizing based on a single context.'''
    def __init__(
            self,
            df: pd.DataFrame = None,
            predictors: dict = None,
            cao_dict: dict = None,
            directions: dict = None,
            parents: list = None,
            method: str = 'nsga2',
            selection_size: int | float | str = None,
            population_size: int = 100,
            n_generations: int = 10,
            elite_ratio: float = 0.1,
            crossover_method: str = 'uniform',
            mutation_rate: float = 0.1,
            divisions: int = 5,
            tournament_size: float | int = 2,
            tournament_replace: bool = True,
            max_grid_points: int = 10000,
            max_attempts: int = 100,
            verbosity: int = DETAILED,
            random_state: int = None
            ):
        if cao_dict is not None:
            if set(cao_dict.keys()) != {'context', 'actions', 'outcomes'}:
                raise ValueError(
                    "Keys of cao_dict has to be 'context', "
                    "'actions', and 'outcomes'"
                )
            for k, v in cao_dict.items():
                if not isinstance(v, list):
                    raise TypeError(f"Value for '{k}' must be a list.")
                if not all(isinstance(item, str) for item in v):
                    raise TypeError(
                        f"All elements in the '{k}' list must be strings.")
        self.cao_dict = cao_dict

        if predictors is not None and cao_dict is not None:
            if not isinstance(predictors, dict):
                raise TypeError(
                    "'Predictors' has to be a dictionary with outcomes "
                    "as keys and Predictor as values"
                )
            if set(predictors.keys()) != set(cao_dict['outcomes']):
                raise ValueError(
                    "Keys in predictors and values of outcomes of "
                    "cao_dict have to be the same"
                )
            # Reorder predictors according to cao_dict['outcomes']
            predictors = {outcome: predictors[outcome]
                          for outcome in cao_dict['outcomes']}
        self.predictors = predictors

        if directions is not None and cao_dict is not None:
            if not isinstance(directions, dict):
                raise TypeError(
                    "'Directions' has to be a dictionary with outcomes as "
                    "keys and 'maximize' or 'minimize' as values"
                )
            if set(directions.keys()) != set(cao_dict['outcomes']):
                raise ValueError(
                    "Keys in directions and values of outcomes of "
                    "cao_dict have to be the same"
                )
            for k, v in directions.items():
                if v not in ['maximize', 'minimize']:
                    raise ValueError(
                        f"Direction of {k} is neither "
                        "'maximize' nor 'minimize'"
                    )
            # Reorder directions according to cao_dict['outcomes']
            directions = {outcome: directions[outcome]
                          for outcome in cao_dict['outcomes']}

        self.directions = directions
        self.directions_list = None
        self.df = df
        self.parents = parents
        self.method = method
        self.selection_size = selection_size
        self.population_size = population_size
        self.n_generations = n_generations
        self.elite_ratio = elite_ratio
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.divisions = divisions
        self.tournament_size = tournament_size
        self.tournament_replace = tournament_replace
        self.max_grid_points = max_grid_points
        self.max_attempts = max_attempts
        self.verbosity = verbosity
        self.random_state = random_state

        self.best_trials = None
        self.context_df = None
        self.pareto_df = None

    def _pareto(self, population, fitnesses) -> pd.DataFrame:
        fronts = non_dominated_sorting(fitnesses, self.directions_list)
        actions_df = pd.DataFrame([population[i] for i in fronts[0]],
                                  columns=self.cao_dict['actions'])
        outcomes_df = pd.DataFrame([fitnesses[i] for i in fronts[0]],
                                   columns=self.cao_dict['outcomes'])
        pareto_df = pd.concat([actions_df, outcomes_df], axis=1)
        # Order the DataFrame by the first outcome column
        first_outcome_col = self.cao_dict['outcomes'][0]
        pareto_df = pareto_df\
            .sort_values(by=first_outcome_col).drop_duplicates()

        return pareto_df.reset_index(drop=True)

    def _predict(self, ca_df: pd.DataFrame):
        y_pred = {}
        for outcome, predictor in self.predictors.items():
            y_pred[outcome] = predictor.predict(ca_df)
        return pd.DataFrame(y_pred)

    def initialize_population(self, population_size: int) -> list:
        """Initialize a population with a given population size or parents."""
        population = []
        if self.parents is not None:
            if len(self.parents) <= population_size:
                population.extend(self.parents)
            else:
                ind = np.random.choice(range(len(self.parents)),
                                       population_size, replace=False)
                population.extend([self.parents[i] for i in ind])

        for _ in range(population_size-len(population)):
            individual = self.generate_individual()
            population.append(individual)

        return population

    def generate_individual(self):
        individual = []
        for action in self.cao_dict['actions']:
            column = self.df[action]
            if column.dtype == 'float':
                individual.append(np.random.rand() *
                                  (column.max() - column.min()) + column.min())
            else:
                individual.append(np.random.choice(column.unique()))
        return individual

    def fitness_function(self,
                         population: list[list[float]]
                         ) -> list[list[float]]:
        # Create a DataFrame from the entire population at once
        action_df = pd.DataFrame(population, columns=self.cao_dict['actions'])

        # Concatenate action_df with the context_df for all individuals
        ca_df = pd.concat([pd.concat([self.context_df]*len(population),
                                     ignore_index=True), action_df], axis=1)

        # Collect predictions for the entire population at once
        y_preds = []
        for predictor in self.predictors.values():
            # Make batch predictions for the entire ca_df
            y_preds.append(predictor.predict(ca_df))

        # Transpose y_preds to get fitness for each individual
        return np.array(y_preds).T.tolist()

    def _evo(self, method: str):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_genes = self.df[self.cao_dict['actions']].shape[1]
        mutation_replace = n_genes*[True]
        mutation_specs = []
        for action in self.cao_dict['actions']:
            column = self.df[action]
            if column.dtype == 'float':
                mutation_spec = {'method': 'uniform',
                                 'param': (column.min(), column.max())}
            else:
                mutation_spec = {'method': 'choice',
                                 'param': column.unique()}

            mutation_specs.append(mutation_spec)

        population, fitnesses = self.evolve(self.directions,
                                            method,
                                            self.selection_size,
                                            self.population_size,
                                            self.n_generations,
                                            self.elite_ratio,
                                            self.crossover_method,
                                            self.mutation_rate,
                                            mutation_replace,
                                            mutation_specs,
                                            self.divisions,
                                            self.tournament_size,
                                            self.tournament_replace,
                                            self.max_attempts,
                                            self.verbosity,
                                            )

        return self._pareto(population, fitnesses)

    def _grid(self):
        action_df = self.df[self.cao_dict['actions']]
        # At least 11 points per numeric column
        n_points_number = max(1, 11**action_df
                              .select_dtypes(include='number').shape[1])
        n_points_object = action_df\
            .select_dtypes(exclude='number').nunique().product()
        n_points_total = n_points_number*n_points_object
        if n_points_total > self.max_grid_points:
            raise ValueError(
                "The number of points in searching are "
                f"{n_points_total} > max_grid_points ({self.max_grid_points})."
            )
        combinations = []
        for action in self.cao_dict['actions']:
            column = self.df[action]
            if column.dtype == 'float':
                points = \
                    int((self.max_grid_points/n_points_object)
                        ** (1/len(self.cao_dict['actions'])))
                combinations.append(
                    np.linspace(column.min(), column.max(), points))
            else:
                combinations.append(column.unique())

        population = list(product(*combinations))
        fitnesses = self.fitness_function(population)
        return self._pareto(population, fitnesses)

    def _tpe(self):
        def objective(trial):
            actions_list = []
            for action in self.cao_dict['actions']:
                dtype = self.df[action].dtype
                if dtype == 'float':
                    actions_list.append(trial
                                        .suggest_float(action,
                                                       self.df[action].min(),
                                                       self.df[action].max()))
                elif dtype in ['object', 'bool']:
                    actions_list.append(trial
                                        .suggest_categorical(action,
                                                             self.df[action]
                                                             .unique()))
                else:
                    raise TypeError(
                        'dtype of actions has to be either '
                        '"float", "object" or "bool".'
                    )

            actions_df = pd.DataFrame([actions_list],
                                      columns=self.cao_dict['actions'])
            ca_df = pd.concat([self.context_df, actions_df], axis=1)
            y_pred = []
            for predictor in self.predictors.values():
                y_pred.append(predictor.predict(ca_df))

            return y_pred

        if self.verbosity <= SUMMARY:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        else:
            optuna.logging.set_verbosity(optuna.logging.INFO)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(directions=self.directions_list,
                                    sampler=sampler)
        if self.parents is not None:
            for individual in self.parents:
                init_actions = dict(zip(self.cao_dict['actions'], individual))
                study.enqueue_trial(init_actions)
            n_trials = self.population_size + len(self.parents)
        else:
            n_trials = self.population_size
        study.optimize(objective, n_trials=n_trials)
        self.best_trials = study.best_trials
        # Create list of dictionaries of actions
        actions_dict_list = [i.params for i in self.best_trials]
        # Combine into one dictionary with list of actions
        actions_dict = {key: [d[key] for d in actions_dict_list]
                        for key in actions_dict_list[0]}
        actions_df = pd.DataFrame(actions_dict)

        # Create array of outcomes
        outcomes_array = np.array([trial.values for trial in self.best_trials])
        # Create DataFrame with appropriate outcome columns
        outcomes_df = pd.DataFrame(outcomes_array,
                                   columns=self.cao_dict['outcomes'])

        pareto_df = pd.concat([actions_df, outcomes_df], axis=1)

        # Order the DataFrame by the first outcome column
        first_outcome_col = self.cao_dict['outcomes'][0]
        pareto_df = pareto_df.sort_values(by=first_outcome_col)

        return pareto_df.drop_duplicates().reset_index(drop=True)

    def _hybrid(self):

        pareto_dfs = []
        for method in ['tpe', 'nsga2', 'nsga3',
                       'tournament_nsga2', 'tournament_nsga3']:
            if method == 'tpe':
                pareto_dfs.append(self._tpe())
            else:
                pareto_dfs.append(self._evo(method))

        new_df = pd.concat(pareto_dfs, ignore_index=True)
        fronts = non_dominated_sorting(
            new_df[self.cao_dict['outcomes']].to_numpy(), self.directions_list)

        combined_pareto_df = new_df.iloc[fronts[0]]
        # Order the DataFrame by the first outcome column
        first_outcome_col = self.cao_dict['outcomes'][0]
        combined_pareto_df = \
            combined_pareto_df\
            .sort_values(by=first_outcome_col).drop_duplicates()
        return combined_pareto_df.reset_index(drop=True)

    def prescribe_predict(self,
                          context_df: pd.DataFrame
                          ):
        '''Optimize based on a single context'''
        self.context_df = context_df
        self.directions_list = list(self.directions.values())
        # Make sure that parents has no duplicates
        if self.parents is not None:
            self.parents = list(map(list, set(map(tuple, self.parents))))
        if self.method in ['nsga2', 'nsga3',
                           'tournament_nsga2', 'tournament_nsga3']:
            pareto_df = self._evo(self.method)
        elif self.method == 'grid':
            pareto_df = self._grid()
        elif self.method == 'hybrid':
            pareto_df = self._hybrid()
        elif self.method == 'tpe':
            pareto_df = self._tpe()
        else:
            raise ValueError('Please select valid a method.')
        self.pareto_df = pareto_df
        if self.verbosity != SILENCE:
            print(self.pareto_df)

    def save(self, file_path: str):
        '''Save pareto_df'''
        if self.pareto_df is not None:
            self.pareto_df.to_csv(file_path + '.csv', index=False)
        else:
            raise ValueError("Please call prescribe_predict method first")

    def load(self, file_path: str):
        '''Load pareto_df'''
        self.pareto_df = pd.read_csv(file_path + '.csv')
