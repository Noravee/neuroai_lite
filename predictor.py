'''Combination of DataEncoder and predictor'''

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline


class Predictor(BaseEstimator, RegressorMixin):
    """A custom predictor class that fits a model using a pipeline with
    optional encoding, based on given data and
    context-actions-outcomes dictionary.

    Attributes:
        model (BaseEstimator): The base estimator model to be used
        for prediction.
        outcome (str): Outcome of the predictor
        encoder (TransformerMixin): An optional transformer for encoding data.
        pipeline (Pipeline): A pipeline object to preprocess and
        predict using the model.

    Methods:
        fit(X, y): Fits the model to the input data X and target y
        after preprocessing.
        predict(X): Predicts the target variable for input data X
        after preprocessing.
    """
    def __init__(
            self,
            model: BaseEstimator,
            cao_dict: dict,
            outcome: str,
            encoder: TransformerMixin | None = None,
                ):
        self.model = model
        self.encoder = encoder
        if set(cao_dict.keys()) != {'context', 'actions', 'outcomes'}:
            raise ValueError("""Keys of cao_dict has to be
                             'context', 'actions', and 'outcomes'""")
        for k, v in cao_dict.items():
            if not isinstance(v, list):
                raise TypeError(f"Value for '{k}' must be a list.")
            if not all(isinstance(item, str) for item in v):
                raise TypeError(f"""All elements in the '{k}'
                                list must be strings.""")
        self.cao_dict = cao_dict
        if outcome not in cao_dict['outcomes']:
            raise ValueError(f"{outcome} has to be in {cao_dict['outcomes']}")
        self.outcome = outcome
        steps = [('preprocessor', encoder), ('predictor', model)]
        self.pipeline = Pipeline(steps=steps)

    def fit(self,
            x: pd.DataFrame | np.ndarray = None,
            y: pd.Series | np.ndarray = None,
            df: pd.DataFrame = None
            ):
        '''
        Fits the model to the input data X and target y after preprocessing.
        If X and y are provided, it validates the input data against
        the context and actions defined in the cao_dict.
        If df is provided, it extracts X and y from the DataFrame.
        Finally, it fits the pipeline with the preprocessed data and returns
        the updated Predictor instance.
        '''

        context_action = self.cao_dict['context'] + self.cao_dict['actions']
        # Ensure that either X and y are provided or df is provided,
        # but not both
        if (x is not None or y is not None) and df is not None:
            raise ValueError("Provide either X and y, or df, but not both.")
        if x is None and y is None and df is None:
            raise ValueError("Provide either X and y, or df.")

        # If X and y are provided
        if x is not None and y is not None:
            if isinstance(x, pd.DataFrame) and \
                    set(x.columns) != set(context_action):
                raise ValueError("""Columns of X do not match that of cao
                                 context and actions.""")
            if isinstance(y, pd.Series) and y.name != self.outcome:
                raise ValueError("Name of y does not match with outcome.")
            if isinstance(x, np.ndarray):
                print("""Warning: Input has no columns' name. Assume that
                        columns are arranged in the same way as cao_dict.""")
                if x.shape[1] != len(context_action):
                    raise ValueError("""Number of columns of X does not match
                                     those of cao context and actions.""")

        # If df is provided
        elif df is not None:
            x = df[context_action]
            y = df[self.outcome]

        self.pipeline.fit(x, y)

        return self

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        '''
        Predicts the target variable for input data X after preprocessing.
        Raises a TypeError if the input is not a DataFrame or
        a 2-D numpy array.
        '''
        if isinstance(x, np.ndarray) and x.shape[1] == \
                len(self.cao_dict['context'] + len(self.cao_dict['actions'])):
            print("""Warning: Input has no columns' name. Assume that
                  columns are arranged in the same way as cao_dict.""")
        elif isinstance(x, pd.DataFrame):
            x = x[self.cao_dict['context'] + self.cao_dict['actions']]
        else:
            raise TypeError("""Input has to be either DataFrame or
                            2-D numpy array.""")
        y_pred = self.pipeline.predict(x)

        return y_pred
