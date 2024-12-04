'''TargetEncoder and ColumnTransformer that can inverse transform'''

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, TargetEncoder


class CustomTargetEncoder(TargetEncoder):
    """Custom target encoder with additional methods for fitting, transforming,
    and inverting transformations.

    Extends the TargetEncoder class and includes methods for fitting,
    transforming, and inverting transformations.
    Also contains a function to find the nearest key in a dictionary.
    """
    def __init__(self,
                 categories='auto',
                 target_type='auto',
                 smooth='auto',
                 cv=5,
                 shuffle=True,
                 random_state=None,
                 ):
        super().__init__(categories=categories,
                         target_type=target_type,
                         smooth=smooth,
                         cv=cv,
                         shuffle=shuffle,
                         random_state=random_state)
        self.x_colnames = None

    def scale_avg(self, y):
        '''
        Scale the input data 'y' using Min-Max scaling and calculate
        the average across rows. If 'y' is a pandas Series, convert it
        to a DataFrame before scaling.
        Return the average values of the scaled data.
        '''

        if isinstance(y, pd.Series):
            y = y.to_frame()
        y_scaled = MinMaxScaler().fit_transform(y)
        y_avg = y_scaled.mean(axis=1)
        return y_avg

    def fit(self, X, y):
        self.x_colnames = X.columns
        y_avg = self.scale_avg(y)
        super().fit(X, y_avg)
        return self

    def fit_transform(self, X, y):
        self.x_colnames = X.columns
        y_avg = self.scale_avg(y)
        x_transformed = super().fit_transform(X, y_avg)
        return x_transformed

    def inverse_transform(self, x_transformed):
        '''Inverse the transform using nearest points'''
        x_transformed = np.array(x_transformed)
        original_x = []

        for i, (encodings, categories) in enumerate(zip(self.encodings_,
                                                        self.categories_)):
            # Apply the find_nearest function
            nearest_categories = self.find_nearest(x_transformed[:, i],
                                                   np.array(encodings),
                                                   np.array(categories))
            original_x.append(nearest_categories)

        return pd.DataFrame(np.array(original_x).T, columns=self.x_colnames)

    # Function to find the nearest key in the dictionary
    def find_nearest(self, array, keys, values):
        '''Calculate the nearest points using L1'''
        indices = np.abs(keys - array[:, None]).argmin(axis=1)
        return values[indices]


class CustomColumnTransformer(ColumnTransformer):
    """
    ColumnTransformer that has inverse_transform method which rearranges
    the columns according to the original
    """
    def __init__(self,
                 transformers,
                 remainder='drop',
                 sparse_threshold=0.3,
                 n_jobs=None,
                 transformer_weights=None,
                 verbose=False,
                 verbose_feature_names_out=True
                 ):
        super().__init__(transformers,
                         remainder=remainder,
                         sparse_threshold=sparse_threshold,
                         n_jobs=n_jobs,
                         transformer_weights=transformer_weights,
                         verbose=verbose,
                         verbose_feature_names_out=verbose_feature_names_out,
                         force_int_remainder_cols=False
                         )

    def inverse_transform(self, x_t):
        """
        Apply inverse transformations to the data,
        including passthrough columns.
        """
        x_reconstructed = []
        # Loop through each transformer
        for name, trans, columns in self.transformers_:
            # Find the index of these columns in the transformed data
            transformed_indices = self.output_indices_[name]

            # If transformer are 'drop' or has 0 column then continue to
            # the next transformer
            if trans == 'drop' or len(columns) == 0:
                continue  # Skip 'drop' transformers

            # Check if the transformer supports inverse_transform
            if hasattr(trans, 'inverse_transform'):
                # Apply the inverse transform to these columns
                inversed_arr = trans.inverse_transform(
                    x_t[:, transformed_indices])
                inversed_df = pd.DataFrame(inversed_arr, columns=columns)
                x_reconstructed.append(inversed_df)
            else:
                raise ValueError(f"""Transformer '{name}' does not support
                                 inverse_transform.""")

            x = pd.concat(x_reconstructed, axis=1)
            # Rearragne the columns according to the original excluding
            # the ones that were dropped
            rearranged_columns = \
                [col for col in self.feature_names_in_ if col in x.columns]

        return x[rearranged_columns]
