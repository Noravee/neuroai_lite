import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import streamlit as st
from xgboost import XGBRegressor

from encoder import DataEncoder
from predictor import Predictor
from prescriptor import NeuralNetworkPrescriptor, SingleSampleContextPrescriptor
from pipeline import PredictorPrescriptorPipeline

st.title("NeuroAI Lite")

data_loader_tab, data_profiler_tab, predictor_tab, prescriptor_tab, optimizer_tab = st.tabs(["Data Loader", "Data Profiler", "Predictor", "Prescriptor", "Optimizer"])

df = None
cao = {}
cao_dict = {}
directions = {}
correct_cao = False

with data_loader_tab:
    # Add file uploader
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "txt"])

    # Process the uploaded file
    if uploaded_file is not None:
        try:
            # Attempt to load as a CSV or Excel file
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".txt"):
                df = pd.read_csv(uploaded_file, delimiter="\t")  # Assuming tab-separated
            else:
                st.error("Unsupported file format!")

            # Display the data
            if df is not None:
                st.write("Preview of the uploaded data:")
                st.dataframe(df.head(), hide_index=True)
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")
    else:
        st.info("Please upload a file to get started.")

with data_profiler_tab:
    if df is not None:
        new_df = df.copy()
        col1, col2, col3 = st.columns(3)
        for col in df.columns:
            with col1:
                index = 0
                if df[col].dtype not in ['int', 'float']:
                    index = 1
                data_type = st.selectbox(
                    label=f"{col}",
                    options=("float", "object"),
                    index=index
                    )
                new_df[col] = new_df[col].astype(data_type)
            with col2:
                cao[col] = st.selectbox(label='cao', options=['context', 'actions', 'outcomes'], key='cao_'+col, label_visibility='hidden')
            with col3:
                disabled = False if cao[col] == 'outcomes' else True
                direction = st.selectbox(label='directions', options=['maximize', 'minimize'], key='directions_'+col, label_visibility='hidden', disabled=disabled)
                if not disabled:
                    directions[col] = direction

        for key, value in cao.items():
            cao_dict.setdefault(value, []).append(key)

        correct_cao = True

        missing_keys = {'context', 'actions', 'outcomes'} - set(cao_dict.keys())
        if missing_keys != set():
            correct_cao = False
            st.write(f"CAO does not contain {missing_keys}.")

with predictor_tab:
    if correct_cao:
        st.header('Data encoders')
        scaler = st.selectbox(label='Numeric encoder', options=['MinMaxScaler', 'RobustScaler', 'StandardScaler', None])
        encoder = st.selectbox(label='Object encoder', options=['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder'])

        st.header('Predictors')
        predictors_dict = {}
        X = new_df.drop(cao_dict['outcomes'], axis=1)
        for outcome in cao_dict['outcomes']:
            num_scaler = None
            if scaler is not None:
                num_scaler = getattr(sklearn.preprocessing, scaler)()
            obj_encoder = getattr(sklearn.preprocessing, encoder)()
            data_encoder = DataEncoder(num_transformer=num_scaler, obj_transformer=obj_encoder, bool_transformer=None)
            model_name = st.selectbox(label=f'predictor for {outcome}', options=['RandomForest', 'XGBoost'])
            model = RandomForestRegressor if model_name == 'RandomForest' else XGBRegressor()
            predictor = Predictor(model=model, encoder=data_encoder, cao_dict=cao_dict, outcome=outcome)
            predictors_dict[outcome] = predictor

        st.header('Check CV score')
        metric = st.selectbox(label='Select metric', options=['MAE', 'MSE', 'RMSE'])
        if metric == 'MAE':
            scoring = 'neg_mean_absolute_error'
        elif metric == 'MSE':
            scoring = 'neg_mean_squared_error'
        else:
            scoring = 'neg_root_mean_squared_error'
        if st.button("Do cross-validation"):
            for outcome, predictor in predictors_dict.items():
                score = cross_val_score(predictor, X, new_df[outcome], scoring=scoring)
                st.write(f'cv score for {outcome}: {-score}')
                st.write(f'mean: {-score.mean()}')
                st.write(f'std: {score.std()}')

contain_prescriptor = False
with prescriptor_tab:
    if correct_cao:
        st.header('Neural network prescriptor')
        nn_hyperparams = {}
        nn_hyperparams['method'] = st.selectbox(label='optimization method', options=['nsga2', 'nsga3', 'tournament_nsga2', 'tournament_nsga3', None], index=4)
        with st.expander('NN prescriptor hyperparameters'):
            nn_tab, evolution_tab = st.tabs(['Neural Network', 'Evolution'])
            with nn_tab:
                nn_hyperparams['n_hidden_layer'] = st.number_input(label='number of hidden layers', min_value=1, step=1, format='%d')
                nn_hyperparams['evolve_activations'] = st.selectbox(label='whether or not to evolve activation function', options=[True, False])
            with evolution_tab:
                nn_hyperparams['population_size'] = st.number_input(label='population size', value=100, help='number of individuals for evolution optimization')
                nn_hyperparams['n_generations'] = st.number_input(label='number of generations', value=10, help='optimize for number of generations')
                nn_hyperparams['elite_ratio'] = st.slider(label='elite ratio', min_value=0.0, max_value=1.0, value=0.1, help='percentage of individuals passed to the next generation without mutation')
                nn_hyperparams['mutation_rate'] = st.slider(label='mutation rate', min_value=0.0, max_value=1.0, value=0.1, help='probability of mutation')
            nn_random = st.radio("random state:", options=["None", "seed number"])
            if nn_random == "seed number":
                nn_hyperparams['random_state'] = st.number_input(label='seed number', min_value=0, step=1, format="%d")
            else:
                nn_hyperparams['random_state'] = None

        if nn_hyperparams['method'] is not None:
            nn = NeuralNetworkPrescriptor(**nn_hyperparams)
            contain_prescriptor = True
        else:
            nn = None

        st.header('Single sample context prescriptor')
        ssc_hyperparams = {}
        ssc_hyperparams['method'] = st.selectbox(label='optimization method', options=['grid', 'nsga2', 'nsga3', 'tournament_nsga2', 'tournament_nsga3', 'tpe', None], index=4)
        with st.expander('hyperparameters'):
            evo_tab, grid_tab = st.tabs(['Evolution & TPE', 'Grid'])
            with evo_tab:
                ssc_hyperparams['population_size'] = st.number_input(label='population size', value=100, help='number of individuals for evolution optimization and number of trials for tpe')
                ssc_hyperparams['n_generations'] = st.number_input(label='number of generations', value=10, help='optimize for number of generations', key='ssc n gen')
                ssc_hyperparams['selection_size'] = st.slider(label='selection size', min_value=0.0, max_value=1.0, value=0.5, help='percentage of individuals to be parents')
                if ssc_hyperparams['selection_size'] in [0.0, 1.0]:
                    ssc_hyperparams['selection_size'] = None
                ssc_hyperparams['elite_ratio'] = st.slider(label='elite ratio', min_value=0.0, max_value=1.0, value=0.1, help='percentage of individuals passed to the next generation without mutation', key='ssc elite')
                ssc_hyperparams['mutation_rate'] = st.slider(label='mutation rate', min_value=0.0, max_value=1.0, value=0.1, help='probability of mutation', key='ssc mutate')
            with grid_tab:
                ssc_hyperparams['max_grid_points'] = st.number_input(label='maximum grid points', value=10000, help='number of searching points')
            ssc_random = st.radio("random state:", options=["None", "seed number"], key='ssc')
            if ssc_random == "seed number":
                ssc_hyperparams['random_state'] = st.number_input(label='seed number', min_value=0, step=1, format="%d", key='ssc random')
            else:
                ssc_hyperparams['random_state'] = None

        if ssc_hyperparams['method'] is not None:
            scp = SingleSampleContextPrescriptor(**ssc_hyperparams)
            contain_prescriptor = True
        else:
            scp = None

        if not contain_prescriptor:
            st.write('At least one of the prescriptor must not be None.')

        pipeline = PredictorPrescriptorPipeline(
            predictors=predictors_dict,
            nn_prescriptor=nn,
            single_prescriptor=scp,
            directions=directions,
            )

def reset(df: pd.DataFrame):
    selectors_filters = [col+'selector' for col in df.columns]
    selectors_filters.append('filters')
    for key in selectors_filters:
        if key in st.session_state:
            del st.session_state[key]

with optimizer_tab:
    if correct_cao and contain_prescriptor:
        st.write('Please select data for context')
        context_dict = {}
        # Create three columns
        cols = st.columns(3)
        for i, col in enumerate(new_df[cao_dict['context']].columns):
            with cols[i % 3]:
                if new_df[col].dtype == 'object':
                    context_dict[col] = [st.selectbox(label=f'{col}', options=new_df[col].unique().tolist())]
                else:
                    context_dict[col] = [st.slider(label=f'{col}', min_value=new_df[col].min(), max_value=new_df[col].max())]

        st.header('Context')
        context_df = pd.DataFrame(context_dict)
        st.dataframe(context_df, hide_index=True)
        if st.button('Fit and Optimize'):
            pipeline.fit(new_df)
            pipeline.prescribe_predict(context_df)
            pareto_df = pipeline.pareto_df

            # Store the DataFrame in the session state to persist between reruns
            st.session_state.df = pareto_df
            st.session_state.filtered_df = pareto_df
            reset(st.session_state.df)

    # After the form is submitted, display the dropdown outside the form
    if 'df' in st.session_state:
        # Initialize session state for filters if not already set
        if 'filters' not in st.session_state:
            st.session_state.filters = {
                i: (st.session_state.df[i].min(), st.session_state.df[i].max()) 
                if st.session_state.df[i].dtype == 'float' 
                else sorted(st.session_state.df[i].unique().tolist()) 
                for i in st.session_state.df.columns
                }

        def filter_df():
            filtered_df = st.session_state.df.copy()
            for col in filtered_df.columns:
                    if filtered_df[col].dtype == 'float':
                        filtered_df = filtered_df.loc[
                            (filtered_df[col] >= st.session_state.filters[col][0]) & 
                            (filtered_df[col] <= st.session_state.filters[col][1])
                            ]
                    else:
                        filtered_df = filtered_df.loc[filtered_df[col].isin(st.session_state.filters[col])]
            return filtered_df

        def generate_filter_select(col: str):

            filters_changed = False
            # Filter based on current selection
            filtered_df = filter_df()

            # Update options for A based on current filtered data
            available = sorted(filtered_df[col].unique().tolist())
            # Multi-select for category
            selected = st.multiselect(
                'Select classes of ' + col, 
                options = sorted(st.session_state.df[col].unique().tolist()),  # Always show full range of options
                default = available,
                key = col + '_selector',
                help = 'Please select at least 1 option.'
                )
            if selected == []:
                st.error('Please select at least 1 option.')
                selected = st.session_state.filters[col]
            # Check if filter changed
            if selected != st.session_state.filters[col]:
                st.session_state.filters[col] = selected  # Update session state based on selection
                filters_changed = True

            return filters_changed

        def generate_filter_slider(col: str):

            filters_changed = False
            # Filter based on current selection
            filtered_df = filter_df()
            # Dynamically update range based on filtered data
            min_default, max_default = filtered_df[col].min(), filtered_df[col].max()

            # Slider for float values
            selected = st.slider(
                    'Select range of ' + col, 
                    min_value = st.session_state.df[col].min(), 
                    max_value = st.session_state.df[col].max(), 
                    value = (min_default, max_default),
                    key = col + '_selector'
                    )
            # Check if filter changed
            if selected != st.session_state.filters[col]:
                st.session_state.filters[col] = selected  # Update session state based on selection
                filters_changed = True

            return filters_changed

        with st.expander("Filters"):
            # Create 2 tabs
            actions_tab, outcomes_tab = st.tabs(['Actions', 'Outcomes'])

            changed = []
            with actions_tab:
                st.subheader('Actions')
                for col in st.session_state.df.select_dtypes(include='object').columns:
                    changed.append(generate_filter_select(col))
            with outcomes_tab:
                st.subheader('Outcomes')
                for col in st.session_state.df.select_dtypes(include='number').columns:
                    changed.append(generate_filter_slider(col))
            filters_changed = any(changed)
            if st.button('Reset Filters', type='primary'):
                reset(st.session_state.df)
                st.rerun()

        # Trigger rerun if filters changed
        if filters_changed:
            st.rerun()

        # Display filtered DataFrame
        st.session_state.filtered_df = filter_df()
        st.dataframe(st.session_state.filtered_df, hide_index=True, width=1000)
