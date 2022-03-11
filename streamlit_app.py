import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from flask import Flask, request, render_template
from flask import Flask


app= Flask(__name__)
model_rfc.fit(x_train, train_targets)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = model_rfc.score(x_train, train_targets)

model= pickle.load(open('finalized_model.sav','rb'))

def run():
    menu = ['Home','Bankruptcy Prediction App']
    choice= st.sidebar.selectbox('Menu',menu)
    if choice== 'Home':
      st.title("Corporate Bankruptcy")
      st.header('What is Bankruptcy?')
      st.write('Bankruptcy is the inability of a person or a buusiness to pay their debt.')
      st.write('It involves the sale of property and some other arrangement to pay as mush as possible of the ,oney of a person or business entity owe.')
      st.write('Bankruptcy is a legal proceeding involving a person or business that is unable to repay their outstanding debts.')
      st.write(' The bankruptcy process begins with a petition filed by the debtor, which is most common, or on behalf of creditors,which is less common.')
      st.write('All of the debtors assets are measured and evaluated, and the assets may be used to repay a portion of outstanding debt.')
      st.subheader('About the Machine Learning Model and Developer')
      st.markdown('''### Model Algorithm : Random Forest Classifier''')
      st.markdown('''### Model Training Accuracy: 100%''')
      st.markdown('''### Model Accuracy: 97.43%''')
      st.markdown('''## Model Developed by Mudit Vyas ~ IIT INDORE''')
      st.markdown('''### contact: mvyas29official@gmail.com''')
      
    elif choice== 'Bankruptcy Prediction App':
      img1 =Image.open('unnamed.png')
      img1=img1.resize((250,200))
      st.image(img1,use_column_width=False)
      st.title("Corporate Bankruptcy prediction Model")
      st.subheader("This model will predict if a company would Bankrupt in future or not")
      #Company Name:
      comp_name= st.text_input('Company Name')
    # [(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] 365
      attr5= st.number_input(' [(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] 365')
    #profit on operating activities / financial expenses
      attr27= st.number_input('profit on operating activities / financial expenses')
    #logarithm of total assets
      attr29= st.number_input('logarithm of total assets')
    #operating expenses / total liabilities
      attr34= st.number_input('operating expenses / total liabilities')
    #35 profit on sales / total assets
      attr35=st.number_input('profit on sales / total assets')
    #(current assets - inventory - receivables) / short-term liabilities
      attr40= st.number_input('(current assets - inventory - receivables) / short-term liabilities')
    #(receivables 365) / sales
      attr44= st.number_input('(receivables 365) / sales')
    #(current assets - inventory) / short-term liabilities
      attr46= st.number_input('(current assets - inventory) / short-term liabilities')
    #47 (inventory 365) / cost of products sold
      attr47= st.number_input('(inventory 365) / cost of products sold')
    #50 current assets / total liabilities
      attr50= st.number_input('current assets / total liabilities')
    #56 (sales - cost of products sold) / sales
      attr56= st.number_input('(sales - cost of products sold) / sales')
    #total costs /total sales
      attr58= st.number_input('total costs /total sales')
    #''Attr5', 'Attr27', 'Attr29', 'Attr34', 'Attr35', 'Attr40', 'Attr44',
       #'Attr46', 'Attr47', 'Attr50', 'Attr56', 'Attr58
      features= [[attr5,attr27,attr29,attr34,attr35,attr40,
                attr44,attr46,attr47,attr50,attr56,attr58 ]]
      print(features)
      y = model.predict(features)
      lc = [str(i) for i in y]
      ans = int("".join(lc))
      if st.button("Predict"):
        if ans == 0:
          st.warning("Your Company " + comp_name + ' is in low risk for bankruptcy.')
        else:
          st.success("Your Company " + comp_name + ' is in high risk for bankruptcy.')


          
@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods= ['POST'])
def index():
    data1= request.form['a']
    data2= request.form['b']
    data3= request.form['c']
    data4= request.form['d']
    data5= request.form['e']
    data6= request.form['f']
    data7= request.form['g']
    data8= request.form['h']
    arr = np.array([[data1,data2,data3,data4,data5,data6,data7,data8]])
    pred= model.predict(arr)
    return render_template('after.html', data=pred)
        


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
