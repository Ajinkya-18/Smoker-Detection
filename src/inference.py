from  utils import get_inference_data, load_model

inf_data = get_inference_data()

model = load_model('models/')

y_hat = model.predict(inf_data)

if int(y_hat[0]) == 1:
    print('Result: Smoker')

elif int(y_hat[0]) == 0:
    print('Result: Non-Smoker')




