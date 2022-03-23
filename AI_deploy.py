#loading the packages
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request

app =Flask(__name__)

#load the model
model=load_model(r'C:\Users\hp\Desktop\01ai_project\data\model.h5')
#load the data
stock_data = pd.read_pickle(r'C:\Users\hp\Desktop\01ai_project\data\KLSE_stock.pkl')
#loading and saving stocks name
data = pd.read_csv(r"C:\Users\hp\Desktop\01ai_project\malasiya_stocks.csv")
names = data['Company Name'].tolist()

#-------------spliting data-------------
n_train = int(stock_data.shape[0]*0.8)
stock_train = stock_data.values[:n_train, :]
stock_test = stock_data.values[n_train:, :]
# normalize Stocks data
scaler = MinMaxScaler([0, 1])
stock_train = scaler.fit_transform(stock_train)
stock_test = scaler.fit_transform(stock_test)

@app.route('/')                
def home():   
    return render_template('web.html')
  
@app.route('/load', methods=['POST'])                
def predict(): 
    n = int(request.form.get("n"))    
    stock_train_pred = model.predict(stock_train)
    error_train = np.mean(np.abs(stock_train - stock_train_pred)**2, axis=0)    
    ind = np.argsort(error_train)
    sort_assets_names = np.array(names)[ind.astype(int)]
    return render_template('web1.html',text1='{}'.format(sort_assets_names[0:n]))
  
         
if __name__ == "__main__":
    app.run(debug=True)                  
    