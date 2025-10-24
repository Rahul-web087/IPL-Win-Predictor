import joblib
pipe = joblib.load('pipe.pkl')
#import pickle
import pandas as pd
# Load your trained Logistic Regression pipeline/model
import streamlit as st
#pipe = pickle.load(open('pipe.pkl','rb'))
teams =['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah',  'Mohali', 'Bengaluru']



st.title('Ipl Win Predictor')

col1, col2 = st.columns(2)
with col1:
   batting_teams= st.selectbox('Select the Batting  Team',sorted(teams))
   with col2:
       bowling_teams= st.selectbox('Select the Bowling  Team', sorted(teams))
       selected_city = st.selectbox('Select the City', sorted(cities))
       target = st.number_input('Enter the Target Score')
       col3,col4,col5=st.columns(3)
       with col3:
           score = st.number_input('Enter the Score')
           with col4:
               overs =st.number_input("Overs completed")
               with col5:
                   wickets =st.number_input("Wickets out")
                   if st.button(' Predict Probability'):
                       runs_left =target - score
                       balls_left= 120- (overs*6)
                       wickets=10- wickets
                       crr= score/overs
                       rrr=(runs_left*6)/balls_left
                       # noinspection PyRedeclaration
                       input_df = input_df = pd.DataFrame(
                           {'batting_team': [batting_teams], 'bowling_team': [bowling_teams], 'city': [selected_city],
                            'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
                            'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})
                       st.table(input_df)
                       result =pipe.predict_proba(input_df)
                       st.text(result)
                       loss = result[0][0]
                       win = result[0][1]
                       st.header(batting_teams+ "- " + str(round(win * 100)) + "%")
                       st.header(bowling_teams + "- " + str(round(loss * 100)) + "%")


