import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import numpy as np
import pandas as pd
from DataLoader import DataLoader
from DataEvaluator import DataEvaluator
from GraphicGenerator import GraphicGenerator
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pickle
#from profanity_check import predict, predict_prob
#import joblib


#Fonction pertmettant d'éviter de rééxecuter les datasets entèirement
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Charger le modèle entraîné
with open('gbr.pkl', 'rb') as file:
    regressor = pickle.load(file)

def predict_price(data):  
        prediction = regressor.predict(data)
        return prediction[0]

def main():
    with st.sidebar:
        choose = option_menu("Main Menu", ["Accueil", "Démo","Evaluation","Prédiction","Contact"],
                            icons=['house-fill', 'file-slides','reception-4','play-circle-fill','envelope-fill'],
                            menu_icon="list", default_index=0,
                            styles={
            "container": {"padding": "5!important", "background-color": "#000"},
            "icon": {"color": "#fff", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#f9d1ac"},
            "nav-link-selected": {"background-color": "#FF9633"},
        }
        )

    if choose == "Accueil":
        col1, col2 = st.columns( [0.8, 0.2])
        with col1:               # To display the header text using css style
            st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">A Propos de l\'application</p>', unsafe_allow_html=True)    
        with col2:               # To display brand log
            st.image("medias/keyce.jpeg", width=200)
        
        st.write("Cette application permet de prédire le coût de l'assurance d'une voiture en fonction de plusieurs paramètres tels que le (Pourcentage de conducteurs impliqués dans des collisions mortelles qui faisaient de la vitesse...).\nCette application a été concue avec le langage Python avec les librairies telles que [Streamlit](https://streamlit.io/), [Pycaret](https://pycaret.org/), [Sklearn](https://scikit-learn.org/stable/)... par [Foupoua Mohamed](https://github.com/mawed22) étudiant en Master 1 IABD à Keyce Informatique Yaoundé sous la supervision de Monsieur [Abdouraman Dalil](https://www.linkedin.com/in/abdouraman-bouba-dalil-3916abb7/).\n\n Testez notre application en ligne [ici](#)") 
        col1,col2 = st.columns(2)
        col1.image("medias/pair.jpeg", width=None)
        col2.image("medias/predict.jpeg", width=None) 

    elif choose=='Démo':
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Démo de l’application...</p>', unsafe_allow_html=True)
        st.markdown(
        "**Comment utiliser cette application ?**\n"
        "\n1- Ajoutez votre dataset au format csv;\n"
        "\n2- Evaluez vos données grace aux différents graphiques;\n"
        "\n3- Faites des prédictions en temps réels.\n"
        )
        st.video("https://www.youtube.com/watch?v=j8LSg3s8ElU")
    
    elif choose=='Evaluation':
        #Ajout d'un fichier au format csv 
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Evaluation des données...</p>', unsafe_allow_html=True) 

        # Chargement du fichier
        st.header('Ajoutez votre Dataset')
        dataLoader = DataLoader()
        dataLoader.check_separator()
        file = dataLoader.load_file()

        if file is not None:
            df = dataLoader.load_data(file)

            # Evaluation des données
            st.header('Évaluation des données')
            dataEvaluator = DataEvaluator(df)
            dataEvaluator.show_head()
            dataEvaluator.show_dimensions()
            dataEvaluator.show_columns()

            # Les differents graphique
            plotGenerator = GraphicGenerator(df)
            
            st.sidebar.text("Sélectionnez les types de graphique")
            checked_scatterPlot = st.sidebar.checkbox('ScatterPlot')
            checked_correlationPlot = st.sidebar.checkbox('Correlation')
            checked_pairplot = st.sidebar.checkbox('PairPlot')
            checked_logisticRegPlot = st.sidebar.checkbox('LogisticRegPlot')

            if checked_scatterPlot:
                st.header('Graphiques scatterPlot')
                plotGenerator.scatterplot()
                st.markdown('<hr/>', unsafe_allow_html=True)

            if checked_correlationPlot:
                st.header('Matrix de corrélation')
                plotGenerator.correlationPlot()
                st.markdown('<hr/>', unsafe_allow_html=True)

            if checked_pairplot:
                st.header('Graphiques PairPlot')
                plotGenerator.pairplot()
                st.markdown('<hr/>', unsafe_allow_html=True)

            if checked_logisticRegPlot:
                st.header('Plot L_Regression')
                plotGenerator.logisticRegressionPlot()
                st.markdown('<hr/>', unsafe_allow_html=True)  

        else:
            st.image("medias/image.jpg")

    elif choose=='Prédiction':
         st.markdown(""" <style> .font {
         font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
         </style> """, unsafe_allow_html=True)
         st.markdown('<p class="font">Prédiction des données</p>', unsafe_allow_html=True)
         #Collecte des données pour la prédiction
         st.sidebar.header("Entrez les données pour la prediction")

        #  State = st.sidebar.selectbox('State',(
        #         'Alabama',
        #         'Alaska',
        #         'Arizona',
        #         'Arkansas',
        #         'California',
        #         'Colorado',
        #         'Connecticut',
        #         'Delaware',
        #         'District of Columbia',
        #         'Florida',
        #         'Georgia',
        #         'Hawaii',
        #         'Idaho',
        #         'Illinois',
        #         'Indiana',
        #         'Iowa',
        #         'Kansas',
        #         'Kentucky',
        #         'Louisiana',
        #         'Maine',
        #         'Maryland',
        #         'Massachusetts',
        #         'Michigan',
        #         'Minnesota',
        #         'Mississippi',
        #         'Missouri',
        #         'Montana',
        #         'Nebraska',
        #         'Nevada',
        #         'New Hampshire',
        #         'New Jersey',
        #         'New Mexico',
        #         'New York',
        #         'North Carolina',
        #         'North Dakota',
        #         'Ohio',
        #         'Oklahoma',
        #         'Oregon',
        #         'Pennsylvania',
        #         'Rhode Island',
        #         'South Carolina',
        #         'South Dakota',
        #         'Tennessee',
        #         'Texas',
        #         'Utah',
        #         'Vermont',
        #         'Virginia',
        #         'Washington',
        #         'West Virginia',
        #         'Wisconsin',
        #         'Wyoming'
        #       ))
         Number_billion_miles=st.sidebar.slider('Number of drivers involved in fatal collisions per billion miles',0.0,50.0,5.0)
         Percentage_Speeding=st.sidebar.slider('Percentage Of Drivers Involved In Fatal Collisions Who Were Speeding',0,100,10)
         Percentage_Alcohol=st.sidebar.slider('Percentage Of Drivers Involved In Fatal Collisions Who Were Alcohol-Impaired',0,100,20)
         Percentage_Not_Distracted=st.sidebar.slider('Percentage Of Drivers Involved In Fatal Collisions Who Were Not Distracted',0,100,15)
         Percentage_Accidents=st.sidebar.slider('Percentage Of Drivers Involved In Fatal Collisions Who Had Not Been Involved In Any Previous Accidents',0,100,35)
         Losses_insured_driver=st.sidebar.slider('Losses incurred by insurance companies for collisions per insured driver ($)',50.0,250.0,100.0)

         # Créer un DataFrame avec les nouvelles données d'entrée
         donnee_entree = pd.DataFrame({
            #'State':State,
            'Number_billion_miles':Number_billion_miles,
            'Percentage_Speeding':Percentage_Speeding,
            'Percentage_Alcohol':Percentage_Alcohol,
            'Percentage_Not_Distracted':Percentage_Not_Distracted,
            'Percentage_Accidents':Percentage_Accidents,
            'Losses_insured_driver':Losses_insured_driver
         }, index=[0])

         # Afficher les données d'entrée
         st.subheader('Les données entrées pour la prédiction')
         st.write(donnee_entree)

        # Faire la prédiction du prix
         prediction = predict_price(donnee_entree)
        # Resultat de la prédiction
         st.subheader('Résultat de la prédiction')
         st.write(f'Montant de l\'assurance :  {prediction:,.2f} $')



    elif choose == "Contact":
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Contactez-nous</p>', unsafe_allow_html=True)
        st.markdown(
         "**Veuillez-nous laisser un message**\n"
         "\n Pour tout problème lié à l'utilisation de l'appli;\n"
         "\n Pour toutes suggestions et remarques\n"
        )

        with st.form(key='columns_in_form2',clear_on_submit=True): 
            col1,col2 = st.columns(2)
            Name = col1.text_input(label='Noms')
            Email = col2.text_input(label=' Email')
            Message=st.text_area(label='Message')
            submitted = st.form_submit_button('Envoyer')
            if submitted:
                if Name == "":
                    st.error("Veuillez remplir tous les champs !!")
                elif Email == "":
                    st.error("Veuillez remplir tous les champs !!")
                elif Message  == "":
                    st.error("Veuillez remplir tous les champs !!")
                else:
                  st.success('Merci de nous avoir contactés. Nous vous répondrons dans les plus brefs délais!')
                  st.write('Votre Nom : ', Name)
                  st.write('Votre Email : ', Email)
                  st.write('Votre Message : ', Message)
    st.sidebar.markdown('''
    ---
    © 2023 Created with ❤️ by [Mohamed](https://mawed22.github.io/myfolio/)
    ''')


if __name__=='__main__':
    main()
