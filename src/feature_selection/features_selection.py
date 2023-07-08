# Standard library imports
from typing import List
import time

# Third-party library imports
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
import eli5
from eli5.sklearn import PermutationImportance
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import make_scorer, mean_squared_error, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import type_of_target

# Scikit-learn related imports
from sklearn.linear_model import ElasticNet, LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.feature_selection import RFE

# Other third-party library imports
from mrmr import mrmr_classif, mrmr_regression
from genetic_selection import GeneticSelectionCV


start_time = time.time()

class FeaturesSelection:
    """
    This class is responsible for feature selection. Each method returns the selected features
    """

    def __init__(self, fs_method_1: str,fs_method_2:str, model_type: str, K: int, random_state: int, alpha: float,
                 l1_ratio: float , C: float , n_features_to_select:int,
                 cv: int, verbose: int,
                 scoring: str,
                 max_features: int,
                 n_population: int,
                 crossover_proba: float,
                 mutation_proba: float,
                 n_generations: int,
                 crossover_independent_proba: float,
                 mutation_independent_proba: float,
                 tournament_size: int,
                 n_gen_no_change: int,
                 caching: bool,
                 n_jobs: int):
        """
        :param fs_method_1: the first method of the feature selection
        :param fs_method_2: the second method of the feature selection
        :param model_type: `regression` or `classification`
        :param K: num of cols to choose for mrmr.
        :param random_state.
        :param alpha: is a regularization parameter that controls the strength of the penalty applied to the coefficients for elastic net.
        :param l1_ratio: controls the mix of L1 and L2 penalties in the Elastic Net regularization.
        :param C: Is the inverse of regularization strength for the classification problem for elastic net.
        :param List of the names of the selected features
        :param n_features_to_select: Number of features to select (for rfe, forward or backward).
        :param cv: The cross-validation splitting strategy for GeneticSelectionCV.
        :param verbose: Controls the verbosity of GeneticSelectionCV.
        :param scoring: Scoring parameter for GeneticSelectionCV.
        :param max_features: Maximum number of features to select in GeneticSelectionCV.
        :param n_population: Size of the population in GeneticSelectionCV.
        :param crossover_proba: Crossover probability in GeneticSelectionCV.
        :param mutation_proba: Mutation probability in GeneticSelectionCV.
        :param n_generations: Number of generations for GeneticSelectionCV.
        :param crossover_independent_proba: Probability of an independent crossover in GeneticSelectionCV.
        :param mutation_independent_proba: Probability of an independent mutation in GeneticSelectionCV.
        :param tournament_size: Size of the tournament in GeneticSelectionCV.
        :param n_gen_no_change: Number of generations with no change to terminate GeneticSelectionCV.
        :param caching: Whether to use caching in GeneticSelectionCV.
        :param n_jobs: Number of jobs to run in parallel in GeneticSelectionCV.
        """
        self.fs_method_1 = fs_method_1
        self.fs_method_2 = fs_method_2
        self.model_type = model_type
        self.K = K
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.C = C
        self.n_features_to_select = n_features_to_select
        self.cv = cv
        self.verbose = verbose
        self.scoring = scoring
        self.max_features = max_features
        self.n_population = n_population
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.n_generations = n_generations
        self.crossover_independent_proba = crossover_independent_proba
        self.mutation_independent_proba = mutation_independent_proba
        self.tournament_size = tournament_size
        self.n_gen_no_change = n_gen_no_change
        self.caching = caching
        self.n_jobs = n_jobs

    def mrmr(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        MRMR (Maximum Relevance Minimum Redundancy) selects informative and non-redundant features by ranking them according
         to their relevance to the target variable while minimizing redundancy between the selected features.
        :param X: X - should be completely numerical
        :param y: the target col
        :return: List of selected features (the top K features)
        """

        if self.model_type == "classification":
            print("Running the mrr: ")
            selected_features = mrmr_classif(X=X, y=y, K=self.K)

        elif self.model_type == "regression":
            print("Running the mrr: ")
            selected_features = mrmr_regression(X=X, y=y, K=self.K)

        return selected_features

    def elastic_net(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        Elastic Net is a linear regression model with both L1 and L2 regularization,
        which combines the strengths of Ridge and Lasso regularization to balance feature selection and model complexity.
        :param X: X - should be completely numerical
        :param y: the target col
        :return: List of selected features
        """

        # Instantiate Elastic Net model based on the model_type
        if self.model_type == "regression":
            model = ElasticNet(
                alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.random_state
            )
        elif self.model_type == "classification":
            # The 'saga' solver is required for Elastic Net regularization in logistic regression
            model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=self.l1_ratio,
                C=self.C,
                random_state=self.random_state,
            )

        # Fit the model
        model.fit(X, y)

        # Get the indices of the best features
        # because they have impact on the model's prediction
        best_feature_indices = np.where(model.coef_ != 0)[0]

        # Get the names of the best features
        selected_features = X.columns[best_feature_indices]
        return selected_features

    def rfe(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        Recursive Feature Elimination (RFE) selects features by recursively eliminating features based on their importance or coefficients.
        :param X: X - should be completely numerical
        :param y: the target col
        :return: List of selected features
        """

        if self.model_type == "regression":
            estimator = lgb.LGBMRegressor(random_state=self.random_state)
        elif self.model_type == "classification":
            estimator = lgb.LGBMClassifier(random_state=self.random_state)

        selector = RFE(estimator=estimator, n_features_to_select=self.n_features_to_select)

        selected_features_to_remove = []

        # Iterate over the RFE steps using tqdm
        with tqdm(total=X.shape[1] - self.n_features_to_select, desc="RFE Progress") as pbar:
            for _ in range(X.shape[1] - self.n_features_to_select):
                selector.fit(X, y)
                feature_ranks = selector.ranking_
                worst_feature = np.argmax(feature_ranks)
                selected_features_to_remove.append(X.columns[worst_feature])
                X = X.drop(columns=X.columns[worst_feature])
                pbar.update()

        selected_features = X.columns[~X.columns.isin(selected_features_to_remove)].tolist()

        return selected_features

    def forward_selection(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        Forward feature selection selects features by iteratively adding the most relevant feature to the feature set.
        :param X: X - should be completely numerical
        :param y: the target col
        :param n_features_to_select: the maximum number of features to select
        :return: List of selected features
        """

        selected_features = []
        remaining_features = set(X.columns)

        # Add tqdm progress bar for feature selection
        with tqdm(total=min(X.shape[1], self.n_features_to_select), desc="Forward Selection Progress") as pbar:
            # Iterate until all features are selected or we've reached the n_features_to_select limit
            while remaining_features and len(selected_features) < self.n_features_to_select:
                best_feature = None
                best_score = np.inf


                # Iterate over remaining features and select the one with the best score
                for feature in remaining_features:
                    feature_set = selected_features + [feature]
                    X_subset = X[feature_set]

                    if self.model_type == "regression":
                        model = lgb.LGBMRegressor(random_state=self.random_state)
                        model.fit(X_subset, y)
                        score = mean_squared_error(y, model.predict(X_subset))
                    elif self.model_type == "classification":
                        model = lgb.LGBMClassifier(random_state=self.random_state)
                        model.fit(X_subset, y)
                        score = f1_score(y, model.predict(X_subset))

                    if score < best_score:
                        best_score = score
                        best_feature = feature

                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                pbar.update(1)  # Update progress bar

        return selected_features

    def backward_selection(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        Backward feature selection selects features by iteratively removing the least relevant feature from the feature set.
        :param X: X - should be completely numerical
        :param y: the target col
        :param n_features_to_select: the maximum number of features to keep
        :return: List of selected features
        """
        selected_features = list(X.columns)

        # Add tqdm progress bar for feature selection
        with tqdm(total=max(0, X.shape[1] - self.n_features_to_select), desc="Backward Selection Progress") as pbar:
            # Iterate until we reach the n_features_to_select limit
            while len(selected_features) > self.n_features_to_select:
                scores = []

                for feature in selected_features:
                    feature_set = selected_features.copy()
                    feature_set.remove(feature)
                    X_subset = X[feature_set]

                    if self.model_type == "regression":
                        model = lgb.LGBMRegressor(random_state=self.random_state)
                        model.fit(X_subset, y)
                        score = mean_squared_error(y, model.predict(X_subset))
                    elif self.model_type == "classification":
                        model = lgb.LGBMClassifier(random_state=self.random_state)
                        model.fit(X_subset, y)
                        score = f1_score(y, model.predict(X_subset))

                    scores.append(score)

                worst_feature = selected_features[np.argmax(scores)]

                selected_features.remove(worst_feature)
                pbar.update(1)  # Update progress bar

        return selected_features

    def genetic_selection(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        GeneticSelectionCV uses Genetic Algorithms to find the optimal subset of features.
        :param X: X - should be completely numerical
        :param y: the target col
        :return: List of selected features
        """

        # Instantiate the estimator according to the problem type
        if self.model_type == "classification":
            estimator = LogisticRegression()
            scoring = "accuracy"
        elif self.model_type == "regression":
            estimator = LinearRegression()
            scoring = "neg_mean_squared_error"

        selector = GeneticSelectionCV(estimator,
                                      cv=self.cv,
                                      verbose=self.verbose,
                                      scoring=self.scoring,
                                      max_features=self.max_features,
                                      n_population=self.n_population,
                                      crossover_proba=self.crossover_proba,
                                      mutation_proba=self.mutation_proba,
                                      n_generations=self.n_generations,
                                      crossover_independent_proba=self.crossover_independent_proba,
                                      mutation_independent_proba=self.mutation_independent_proba,
                                      tournament_size=self.tournament_size,
                                      n_gen_no_change=self.n_gen_no_change,
                                      caching=self.caching,
                                      n_jobs=self.n_jobs)
        selector = selector.fit(X, y)

        selected_features = X.columns[selector.support_].tolist()
        return selected_features

    def _apply_fs_method(self, X, y, method):
        if method == "mrmr":
            selected_features = self.mrmr(X, y=y)

        elif method == "elastic_net":
            selected_features = self.elastic_net(X, y=y)

        elif method == "rfe":
            selected_features = self.rfe(X, y=y)

        elif method == "forward_selection":
            selected_features = self.forward_selection(X, y=y)

        elif method == "backward_selection":
            selected_features = self.backward_selection(X, y=y)

        elif method == "genetic_selection":
            selected_features = self.genetic_selection(X, y=y)

        else:
            print("please provide a valid feature selection method")
            return []

        return selected_features

    def feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """
        Train an XGBoost model and use eli5 to compute and visualize feature importance.
        :param X: DataFrame of features
        :param y: Series of target variable
        :return: The feature importance
        """

        # Determine if task is regression or classification
        task = type_of_target(y)

        if 'continuous' in task:
            model = XGBRegressor()
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
        else:
            model = XGBClassifier()
            scorer = make_scorer(f1_score, greater_is_better=True)

        # Fit your model
        model.fit(X, y)

        # Create the eli5 PermutationImportance object and compute importances
        perm = PermutationImportance(model, random_state=1, scoring=scorer).fit(X, y)

        # Print the feature importances
        print(eli5.format_as_text(eli5.explain_weights(perm, feature_names=X.columns.tolist())))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Applying the feature selection method and return all the feature selection parameters
        including the features names.
        """
        self.selected_features_1 = None
        self.selected_features_2 = None

        print(f"Size of X before feature selection: {X.shape}")

        if self.fs_method_1:
            self.selected_features_1 = self._apply_fs_method(X, y, self.fs_method_1)
            print(f'Number of features after {self.fs_method_1}: {len(self.selected_features_1)}')
            X = X[self.selected_features_1]  # filter the data with the selected features
            print(f'Shape of X after {self.fs_method_1}: {X.shape}')

        if self.fs_method_2:
            self.selected_features_2 = self._apply_fs_method(X, y, self.fs_method_2)
            print(f'Number of features after {self.fs_method_2}: {len(self.selected_features_2)}')
            X = X[self.selected_features_2]  # filter the data with the selected features
            print(f'Shape of X after {self.fs_method_2}: {X.shape}')

        self.final_selected_features = X.columns.tolist()

        # Feature Importance
        X_selected = X[self.final_selected_features]
        self.feature_importance(X_selected, y)

        return self

    def transform(self, X: pd.DataFrame):
        """
        Return the filtered dataframe after the feature selection applied
        """
        # Filter the DataFrame based on the selected features from each method, if they are defined
        if self.selected_features_1 is not None:
            X = X[self.selected_features_1]

        if self.selected_features_2 is not None:
            X = X[self.selected_features_2]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The feature selection was done successfully in {elapsed_time:.2f} seconds")

        return X





