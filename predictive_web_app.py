import numpy as np
import pickle
import streamlit as st


#load the saved model
smote_model = pickle.load(open('trained_model.sav', 'rb'))

#creating a function for prediction

def customer_churn_prediction(input_data):
    
    #changging the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = smote_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "The Customer has NOT Churned"
    else:
        return "The Customer has Churned"
    


def main():

    #giving the title
    st.title("Customer Churn Web App")

    #getting the input data from the user
    CreditScore = st.text_input("Customer's Credit Score")
    Age = st.text_input("AgeGroup 18-30")
    Tenure = st.text_input("How long has the customer been with Bank")
    Balance = st.text_input("Customer's account balance")
    NumOfProducts = st.text_input("Number of Product Customer has subscribed to")
    HasCreditCard = st.text_input("Customer has a card, 1 for Yes, 0 for No")
    IsActiveMember = st.text_input("Customer an active member, 1 for Yes, 0 for No")
    EstimatedSalary = st.text_input("Salary of customer")
    Geography_Germany= st.text_input("Customer lives in Germany, 1 for Yes, 0 for No")
    Geography_Spain= st.text_input("Customer lives in Spain, 1 for Yes, 0 for No")
    Gender = st.text_input("1 for Male, 0 for Female")
    AgeGroup_31_45= st.text_input("AgeGroup_31-45, 1 for Yes, 0 for No")
    AgeGroup_46_60 = st.text_input("AgeGroup_46-60, 1 for Yes, 0 for No")
    AgeGroup_60 = st.text_input("AgeGroup_60+, 1 for Yes, 0 for No")
    BalanceBucket_Low = st.text_input("Balance between 0-49,000, 1 for Yes, 0 for No")
    BalanceBucket_Medium = st.text_input("Balance between 50,000 - 99,000, 1 for Yes, 0 for No")
    BalanceBucket_High = st.text_input("Balance between 100,000+, 1 for Yes, 0 for No")


    #code for prediction
    diagnosis = ''

    #creating a button for Prediction
    if st.button("Customer Churn"):
        diagnosis = customer_churn_prediction([CreditScore, Age, Tenure, Balance, NumOfProducts, HasCreditCard, IsActiveMember,
                                               EstimatedSalary, Geography_Germany, Geography_Spain, Gender, AgeGroup_31_45, AgeGroup_46_60, AgeGroup_60, BalanceBucket_Low, BalanceBucket_Medium, BalanceBucket_High])
        

    
    st.success(diagnosis)



if __name__ == '__main__':
    main()



                                     