def assert_path(path:str):
    import os
    from pathlib import Path

    cwd = os.getcwd()
    full_path = Path(os.path.join(cwd, path))

    if (os.path.exists(full_path) and (full_path.suffix=='.csv')) or full_path.suffix=='.joblib':
        return Path(full_path)
    
    else:
        raise ValueError('Invalid file path or file extension.')

#---------------------------------------------------------------------------------------------------------------

def load_data(data_path:str):
    try:
        full_data_path = assert_path(data_path)

        import pandas as pd

        df = pd.read_csv(full_data_path, index_col='id')

        return df
    
    except Exception as e:
        raise e
    
#--------------------------------------------------------------------------------------------------------------

def split_data(df, target:str='smoking'):
    try:
        X, Y = df.drop([target], axis=1), df[target]

        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)

        return x_train, x_val, y_train, y_val
    
    
    except Exception as e:
        raise e
    
#-----------------------------------------------------------------------------------------------------------------

def preprocess_data(df, target:str='smoking', mode:str='train'):
    try:
        df['BMI'] =  df['weight(kg)'] / (df['height(cm)']/100)**2
        df['WHtR'] = df['waist(cm)'] / df['height(cm)']
        df['eyesight'] = df['eyesight(left)'] + df['eyesight(right)'] / 2

        df.drop(['weight(kg)', 'height(cm)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)', 
                     'hearing(left)', 'hearing(right)'], 
                     axis=1, inplace=True)


        if mode=='train':

            x_train, x_val, y_train, y_val = split_data(df, target=target)

            from sklearn.preprocessing import QuantileTransformer
            from sklearn.feature_selection import RFECV
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import StratifiedKFold

            scaler = QuantileTransformer(random_state=42)

            estimator_dtc = DecisionTreeClassifier(max_depth=15, min_samples_split=30, 
                                                   min_samples_leaf=15, random_state=42)
            
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            selector = RFECV(estimator=estimator_dtc, step=1, min_features_to_select=15, 
                             cv=cv, scoring='f1', verbose=1, n_jobs=5, importance_getter='auto')

            x_train_scaled = scaler.fit_transform(x_train)
            x_val_scaled = scaler.transform(x_val)

            x_train_scaled_best = selector.fit_transform(x_train_scaled)
            x_val_scaled_best = selector.transform(x_val_scaled)

            return x_train_scaled_best, x_val_scaled_best, y_train, y_val
        

        if mode=='inference':
            scaler = load_model('models/quantile_transformer.joblib')
            selector = load_model('models/RFECV_fitted.joblib')

            df_scaled = scaler.transform(df)

            df_scaled_best = selector.transform(df_scaled)

            # from random import choice
            # i = choice(range(0, len(df_scaled_best)))

            return df_scaled_best #.iloc[i, :].values.reshape(1, -1)
        

        else:
            raise ValueError('Invalid "mode" value passed.')
        

    except Exception as e:
        raise e

#-----------------------------------------------------------------------------------------------------------------

# Use when best features are saved directly to a joblib file
def get_best_features(best_features_path:str='models/best_features.joblib'):
    try:
        from joblib import load
        with open(assert_path(best_features_path), 'rb') as f:
            best_features = load(f)

            return best_features
        

    except Exception as e:
        raise e

#----------------------------------------------------------------------------------------------------------------

def save_model(model_object, save_path:str):
    try:
        full_save_path = assert_path(save_path)

        from joblib import dump

        print('Saving model...')
        with open(full_save_path, 'wb') as f:
            dump(model_object, f)
        
        print('Model saved successfully!')
        

    except Exception as e:
        raise e

#--------------------------------------------------------------------------------------------------------------------

def load_model(model_path:str):
    try:
        full_model_path = assert_path(model_path)

        from joblib import load
        with open(full_model_path, 'rb') as f:

            # print('Loading model...')
            model = load(f)
            # print('Model loaded successfully!')

            return model
        
    except Exception as e:
        raise e
    
#----------------------------------------------------------------------------------------------------------------

def train_model(model_instance, x_train, y_train):
    try:
        print('Model training started...')
        model_instance.fit(x_train, y_train)
        print('Model training completed! Returning fitted model instance...')

        return model_instance
    
    except Exception as e:
        raise e
    
#------------------------------------------------------------------------------------------------------------------------

def test_model(model_instance, x_test, y_test, metric:str='accuracy'):
    try:
        y_preds = model_instance.predict(x_test)

        if metric=='accuracy':
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y_test, y_preds)

            return score
        
        if metric=='f1_score':
            from sklearn.metrics import f1_score
            score = f1_score(y_true=y_test, y_pred=y_preds)

            return score
        
    
    except Exception as e:
        raise e
    
#----------------------------------------------------------------------------------------------------------


