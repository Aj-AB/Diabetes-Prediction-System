import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('train_model.sav', 'rb'))


def diabetic_predict(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main(): 
    # title
    st.title('Diabetes Prediction System App')
    Pregnancies=st.text_input('Numbers of Prenancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('BloodPressure Level')
    SkinThickness=st.text_input('SkinThickness Level')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function Level')
    Age=st.text_input('Age value')

    diagnosis=''

    if st.button('Diabetes Test Result'):
        diagnosis=diabetic_predict([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
if __name__ =='__main__':
    main()
    

    





