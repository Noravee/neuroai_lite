'''Class on top of CustomColumnTransformer'''

from sklearn.base import BaseEstimator, TransformerMixin

from customsklearn import CustomColumnTransformer


class DataEncoder(BaseEstimator, TransformerMixin):
    """
    A custom transformer class for encoding and preprocessing different
    data types in a machine learning pipeline.

    Attributes:
    - num_transformer: Transformer for numerical data.
    - obj_transformer: Transformer for categorical data.
    - bool_transformer: Transformer for boolean data.
    - preprocessor: ColumnTransformer for preprocessing data based on
    specified transformers.

    Methods:
    - construct_preprocessor: Constructs a ColumnTransformer for preprocessing
    data based on specified transformers.
    - fit: Fit the transformers to the data and target.
    - transform: Apply the transformations to the features.
    - fit_transform: Fit and transform the data.
    - inverse_transform: Perform inverse transformation on the input data.
    """
    def __init__(self,
                 num_transformer=None,
                 obj_transformer=None,
                 bool_transformer=None):
        """
        Initialize the DataEncoder with transformers for each data type and
        drop columns of any data type other than 'number', 'object', and 'bool'

        Parameters:
        - num_transformer: Transformer for numerical data
        (e.g., StandardScaler, RobustScaler, MinMaxScaler).
        - obj_transformer: Transformer for categorical (object) data
        (e.g., OrdinalEncoder, OneHotEncoder, TargetEncoder).
        - bool_transformer: Transformer for boolean data
        (e.g., OrdinalEncoder).
        - y_transformer: Transformer for the target variable y
        (e.g., MinMaxScaler, LabelEncoder).
        """
        self.num_transformer = num_transformer
        self.obj_transformer = obj_transformer
        self.bool_transformer = bool_transformer
        self.preprocessor = None

    def construct_preprocessor(self,
                               num_cols: list[str],
                               obj_cols: list[str],
                               bool_cols: list[str]):
        '''
        Constructs a ColumnTransformer for preprocessing numerical,
        categorical, and boolean columns based on the specified transformers.
        If a transformer is None, the columns are passed through as
        'passthrough'. Returns the constructed preprocessor.
        '''
        transformers = []
        if self.num_transformer is None:
            transformers.append(('num', 'passthrough', num_cols))
        else:
            transformers.append(('num', self.num_transformer, num_cols))
        if self.obj_transformer is None:
            transformers.append(('obj', 'passthrough', obj_cols))
        else:
            transformers.append(('obj', self.obj_transformer, obj_cols))
        if self.bool_transformer is None:
            transformers.append(('bool', 'passthrough', bool_cols))
        else:
            transformers.append(('bool', self.bool_transformer, bool_cols))

        # Create the ColumnTransformer with the transformers
        preprocessor = CustomColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop any other columns not specified
        )

        return preprocessor

    def fit(self, x, y=None):
        """
        Fit the transformers to the data and target (y).
        """
        # Select columns based on data type
        num_cols = x.select_dtypes(include='number').columns
        obj_cols = x.select_dtypes(include='object').columns
        bool_cols = x.select_dtypes(include='bool').columns

        self.preprocessor = self.construct_preprocessor(num_cols,
                                                        obj_cols,
                                                        bool_cols)

        # Fit the preprocessor for X
        self.preprocessor.fit(x, y)

        return self

    def transform(self, x):
        """
        Apply the transformations to the features (X) and optionally to
        the target (y).
        """
        x_transformed = self.preprocessor.transform(x)

        return x_transformed

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit and transform the data and optionally the target variable.
        """
        # Select columns based on data type
        num_cols = X.select_dtypes(include='number').columns
        obj_cols = X.select_dtypes(include='object').columns
        bool_cols = X.select_dtypes(include='bool').columns

        self.preprocessor = self.construct_preprocessor(num_cols,
                                                        obj_cols,
                                                        bool_cols)

        x_transformed = self.preprocessor.fit_transform(X, y)

        return x_transformed

    def inverse_transform(self, x_transformed):
        """
        Perform inverse transformation on the input data
        that was previously transformed.

        Parameters:
        X_transformed (array-like): The transformed input data.

        Returns:
        array-like: The original input data before transformation.
        """
        original_x = self.preprocessor.inverse_transform(x_transformed)
        return original_x
