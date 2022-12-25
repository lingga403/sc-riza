import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

df_toyota = pd.read_csv("dataset/Data toyota.csv", delimiter=';')

df_toyota["Date"]=pd.to_datetime(df_toyota.Date, format="%Y-%m")
df_toyota.index=df_toyota['Date']

data=df_toyota.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_toyota)),columns=['Date','Price'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Price"][i]=data["Price"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values

train=dataset[0:132,:] #ganti di sini
valid=dataset[132:,:] #ganti di sini

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=load_model("model/saved_model.h5")

inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_data[:132] #ganti di sini
valid=new_data[132:] #ganti di sini
valid['Predictions']=closing_price

app.layout = html.Div([
   
    html.H1("Dashboard Prediksi Harga Mobil Toyota Avanza", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Dataset Mobil Toyota Avanza',children=[
            html.Div([
                html.H2("Harga Aktual Mobil Toyota Avanza",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=train.index,
                                y=train["Price"],
                                # mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Periode'},
                            yaxis={'title':'Jumlah Penjualan'}
                        )
                    }
                ),
                html.H2("Prediksi Harga Mobil Toyota Avanza Algoritma LSTM",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid.index,
                                y=valid["Predictions"],
                                # mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Periode'},
                            yaxis={'title':'Jumlah Penjualan'}
                        )
                    }
                )                
            ])                
        ])
    ])
])

if __name__=='__main__':
    app.run_server(debug=True)