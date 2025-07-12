def assert_path(path:str):
    import os
    from pathlib import Path

    cwd = os.getcwd()
    full_path = os.path.join(cwd, Path(path))

    if os.path.exists(full_path) and (full_path.endswith('.csv') or full_path.endswith('.joblib')):
        return Path(full_path)
    
    else:
        raise ValueError('Invalid file path or file extension.')

#---------------------------------------------------------------------------------------------------------------

def load_data(data_path:str):
    try:
        full_data_path = assert_path(data_path)

        import pandas as pd

        df = pd.read_csv(full_data_path)

        return df
    
    except Exception as e:
        raise e
    
#--------------------------------------------------------------------------------------------------------------

def split_data(df, target:str):
    try:
        from sklearn.model_selection import train_test_split

        X, Y = df.drop(['smoking'], axis=1), df['smoking'].values.reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        raise e
    
#-----------------------------------------------------------------------------------------------------------------

def preprocess_data(df, target:str, mode:str='train'):
    try:
        df['BMI'] =  df['weight(kg)'] / (df['height(cm)']/100)**2
        df['WHtR'] = df['waist(cm)'] / df['height(cm)']
        df['eyesight'] = df['eyesight(left)'] + df['eyesight(right)'] / 2
        df['hearing'] = df['hearing(left)'] + df['hearing(right)'] / 2

        df.drop(['weight(kg)', 'height(cm)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)', 
                     'hearing(left)', 'hearing(right)'], axis=1, inplace=True)


        if mode=='test':
            pass

        else:
            raise ValueError('Invalid "mode" value passed.')
        

    except Exception as e:
        raise e

#-----------------------------------------------------------------------------------------------------------------

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

            print('Loading model...')
            model = load(f)
            print('Model loaded successfully!')

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

def test_model(model_instance, x_test, y_test):
    try:
        y_preds = model_instance.predict(x_test)

        from sklearn.metrics import r2_score
        score = r2_score(y_test, y_preds)

        return score
    
    except Exception as e:
        raise e
    
#----------------------------------------------------------------------------------------------------------

def get_inference_data():
    from random import choice, uniform
    import pandas as pd
    
    inf_data = {}

    return pd.DataFrame.from_dict(inf_data)

#----------------------------------------------------------------------------------------------------------


