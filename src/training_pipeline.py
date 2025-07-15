from utils import load_data, preprocess_data, test_model, save_model, train_model


# load data
train_df = load_data('data/train/train.csv')
print(train_df.shape)

# preprocess data
x_train, x_val, y_train, y_val = preprocess_data(train_df, mode='train')
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)


# train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=160, max_depth=10, min_samples_split=30, 
                               min_samples_leaf=15, bootstrap=True, oob_score=True, 
                               class_weight='balanced', n_jobs=5, random_state=42)

trained_model = train_model(model_instance=model, x_train=x_train, y_train=y_train)

# save trained model
save_model(model_object=trained_model, save_path='models/balanced_RFC.joblib')


#test model performance
score = test_model(model_instance=trained_model, x_test=x_val, y_test=y_val, metric='f1_score')

print(f'F1 Score: {score:.2f}')

