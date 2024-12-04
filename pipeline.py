'''Pipeline that combine predictor and prescriptor'''
import pandas as pd


class PredictorPrescriptorPipeline():
    '''
    Pipeline class that combines predictors and prescriptors for fitting,
    predicting, prescribing actions, saving, and loading pareto dataframes.
    '''
    def __init__(self,
                 predictors: dict,
                 nn_prescriptor=None,
                 single_prescriptor=None,
                 directions: dict = None):
        # Ensure all predictors have the same cao_dict
        first_cao_dict = None
        for model in predictors.values():
            if first_cao_dict is None:
                first_cao_dict = model.cao_dict
            elif model.cao_dict != first_cao_dict:
                raise ValueError(
                    "All models in predictors must have the same cao_dict.")
        self.cao_dict = first_cao_dict
        # Check if directions is not None if either prescriptor is provided
        if (nn_prescriptor or single_prescriptor) and directions is None:
            raise ValueError(
                "directions must be provided if nn_prescriptor or "
                "single_prescriptor is specified."
            )

        # Initialize attributes
        self.directions = directions
        self.predictors = predictors
        self.nn_prescriptor = nn_prescriptor
        self.single_prescriptor = single_prescriptor

        self.df = None
        self.avg_pareto_df = None
        self.pareto_df = None

        # Set cao_dict and directions for any non-None prescriptor
        for model in [self.nn_prescriptor, self.single_prescriptor]:
            if model is not None:
                model.cao_dict = self.cao_dict
                model.directions = self.directions

    def fit(self, df: pd.DataFrame):
        '''Fit predictors and prescriptor if available'''
        self.df = df
        self._fit_predictor()
        if self.nn_prescriptor is not None:
            self._fit_prescriptor()
        return self.predictors

    def _fit_predictor(self):
        for outcome, predictor in self.predictors.items():
            x = self.df[predictor.cao_dict['context'] +
                        predictor.cao_dict['actions']]
            y = self.df[outcome]
            predictor.fit(x, y)

    def _fit_prescriptor(self):
        self.nn_prescriptor.predictors = self.predictors
        self.nn_prescriptor.fit(self.df)
        self.avg_pareto_df = self.nn_prescriptor.pareto_df

    def predict(self, df: pd.DataFrame):
        '''Predict outcomes from context and actions'''
        y_pred = {}
        for outcome, predictor in self.predictors.items():
            x = df[predictor.cao_dict['context'] +
                   predictor.cao_dict['actions']]
            y_pred[outcome] = predictor.predict(x)
        return pd.DataFrame(y_pred)

    def prescribe_predict(self, context_df: pd.DataFrame):
        '''Prescribe actions from context and predict outcomes'''
        # Check for prescriptors and handle error if neither is provided
        if self.nn_prescriptor is None and self.single_prescriptor is None:
            raise ValueError("There is no prescriptor.")

        # Set up single_prescriptor if it exists
        if self.single_prescriptor is not None:
            self.single_prescriptor.predictors = self.predictors
            self.single_prescriptor.df = self.df

        # Case: Both prescriptors are present
        if self.nn_prescriptor is not None and\
                self.single_prescriptor is not None:
            parents = self.nn_prescriptor\
                .prescribe_predict_all(context_df)[self.cao_dict['actions']]\
                .to_numpy().tolist()
            self.single_prescriptor.parents = parents
            self.single_prescriptor.prescribe_predict(context_df)
            self.pareto_df = self.single_prescriptor.pareto_df
            return

        # Case: Only single_prescriptor is present
        if self.single_prescriptor is not None:
            self.single_prescriptor.prescribe_predict(context_df)
            self.pareto_df = self.single_prescriptor.pareto_df
            return

        # Case: Only nn_prescriptor is present
        self.nn_prescriptor.prescribe_predict_all(context_df)

    def save(self, file_path: str):
        '''Save pareto df'''
        if self.nn_prescriptor is not None:
            self.nn_prescriptor.save(file_path+'nn')
        if self.single_prescriptor is not None:
            self.single_prescriptor.save(file_path+'single')

    def load(self, file_path: str):
        '''Load pareto df'''
        if self.nn_prescriptor is not None:
            self.nn_prescriptor.load(file_path+'nn')
        if self.single_prescriptor is not None:
            self.single_prescriptor.load(file_path+'single')
