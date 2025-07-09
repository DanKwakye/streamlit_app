import numpy as np
import pickle

#loading the trained model
smote_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (619,42,2,0.00,1,1,1,101348.88,0,0,0,1,0,0,0,1,0)


#changing the data inpute to np array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = smote_model.predict(input_data_reshaped)

print(prediction)

if (prediction[0] == 0):
    print("The Customer has not Churned")
else:
    print("The person has Churned")