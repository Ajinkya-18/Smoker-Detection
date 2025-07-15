from  utils import load_data, load_model, preprocess_data
# import numpy as np

# load data
test_df = load_data('data/test/test.csv')

# preprocess data
df_processed = preprocess_data(test_df, mode='inference')
# print(np.shape(df_processed))

# load trained model
lr = load_model('models/logistic_regression.joblib')
rfc = load_model('models/balanced_RFC.joblib')

# print(lr.predict(df_processed)[0], rfc.predict(df_processed)[0])
model = rfc

y_hat = model.predict(df_processed)

if int(y_hat[0]) == 1:
    print('Result: Smoker')

elif int(y_hat[0]) == 0:
    print('Result: Non-Smoker')




