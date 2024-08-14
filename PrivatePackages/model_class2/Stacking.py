from sklearn.ensemble import StackingClassifier, StackingRegressor


class StackingClassifier_Pro():

    """ A tuneable class wrapper of Sklearn's stacking classifier which compatible
    with JiaXing tuners """

    def __init__(self, **kwargs):

        # get the list of parameters for the final estimator as opposed to the stacker
        final_estimators_params = {key: kwargs[key] for key in kwargs if key not in ['estimators_list', 'num_estimators', 'cv', 'stack_method', 'n_jobs', 'passthrough', 'verbose', 'final_estimator_class']}
        

        self.estimators_list = kwargs['estimators_list']
        self.num_estimators = kwargs['num_estimators']
        self.cv = kwargs['cv']
        self.stack_method = kwargs.get('stack_method', 'auto')
        self.n_jobs = kwargs['n_jobs']
        self.passthrough = kwargs.get('passthrough', False)
        self.verbose = kwargs.get('verbose', 0)
        self.final_estimator = kwargs['final_estimator_class'](**final_estimators_params) # this setup allows to tune the hp of the final estimator

        self.model = StackingClassifier(
            estimators=self.estimators_list[:self.num_estimators], # this setup allows for the user to choose the number of base estimators to use in the stacking model
            final_estimator=self.final_estimator,
            cv=self.cv,
            stack_method=self.stack_method,
            n_jobs=self.n_jobs,
            passthrough=self.passthrough,
            verbose=self.verbose
        )

    def fit(self, X, y):
        """ Fit the model to the training data
        
            Input:
                - X: pd.DataFrame, containing features
                - y: pd.DataFrame, containing targets

        """
        
        self.model.fit(X, y)


    def predict(self, X):
        """ Predict the targets for the test data
        
            Input:
                - X: pd.DataFrame, containing features

            Output:

                - 'y_pred': pd.DataFrame, containing predicted targets

        """

        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)
    
    def set_params(self, **params):
        return self.model.set_params(**params)
    
    def fit_transform(self, X, y):
        return self.model.fit_transform(X, y)

    def get_feature_names_out(self, input_features=None):
        return self.model.get_feature_names_out(input_features=input_features)

    def get_metadata_routing(self):
        return self.model.get_metadata_routing()
    
    def set_fit_request(self, fit_request):
        return self.model.set_fit_request(fit_request)
    
    def transform(self, X):
        return self.model.transform(X)
    
    def decision_function(self, X):
        return self.model.decision_function(X)
    
    def set_fit_request(self, fit_request):
        return self.model.set_fit_request(fit_request)
    
    def set_output(self, output):
        return self.model.set_output(output)


class StackingRegressor_Pro():
    
    """ A tuneable class wrapper of Sklearn's stacking regressor which compatible
    with JiaXing tuners """

    def __init__(self, **kwargs):

        # get the list of parameters for the final estimator as opposed to the stacker
        final_estimators_params = {key: kwargs[key] for key in kwargs if key not in ['estimators_list', 'num_estimators', 'cv', 'n_jobs', 'passthrough', 'verbose', 'final_estimator_class']}

        self.estimators_list = kwargs['estimators_list']
        self.num_estimators = kwargs['num_estimators']
        self.cv = kwargs['cv']
        self.n_jobs = kwargs['n_jobs']
        self.passthrough = kwargs.get('passthrough', False)
        self.verbose = kwargs.get('verbose', 0)
        self.final_estimator = kwargs['final_estimator_class'](final_estimators_params) # this setup allows to tune the hp of the final estimator

        self.model = StackingRegressor(
            estimators=self.estimators_list[:self.num_estimators],
            final_estimator=self.final_estimator,
            cv=self.cv,
            n_jobs=self.n_jobs,
            passthrough=self.passthrough,
            verbose=self.verbose
        )

    def fit(self, X, y):
        """ Fit the model to the training data
        
            Input:
                - X: pd.DataFrame, containing features
                - y: pd.DataFrame, containing targets

        """

        self.model.fit(X, y)


    def predict(self, X):
        """ Predict the targets for the test data
        
            Input:
                - X: pd.DataFrame, containing features

            Output:

                - 'y_pred': pd.DataFrame, containing predicted targets

        """

        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)
    
    def set_params(self, **params):
        return self.model.set_params(**params)
    
    def fit_transform(self, X, y):
        return self.model.fit_transform(X, y)

    def get_feature_names_out(self, input_features=None):
        return self.model.get_feature_names_out(input_features=input_features)

    def get_metadata_routing(self):
        return self.model.get_metadata_routing()
    
    def set_fit_request(self, fit_request):
        return self.model.set_fit_request(fit_request)
    
    def transform(self, X):
        return self.model.transform(X)
    
    def set_fit_request(self, fit_request):
        return self.model.set_fit_request(fit_request)
    
    def set_output(self, output):
        return self.model.set_output(output)
    