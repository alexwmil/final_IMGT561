import requests
import csv
import database
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from itertools import product
import os
import time
import numpy as np

def get_data_and_make_db(symbols: list,database_paths: list, slices: list):
    for symbol,database_path in zip(symbols, database_paths):

        my_list = []
        for slice in slices:
            CSV_URL = "https://www.alphavantage.co/query?function=" + function + "&symbol=" + symbol + "&interval=" + interval + "&slice=" + slice + "&apikey=" + key
            with requests.Session() as s:
                download = s.get(CSV_URL)
                decoded_content = download.content.decode('utf-8')
                cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                cr = list(cr)
                #remove col names from list of entries
                del cr[0]
                my_list = my_list + cr

        #initialize DB    
        db = database.Database(database_path)

        
        #insert rows into DB
        if insert:
            for i in range(len(my_list)):
                if len(my_list[i]) == 6:
                    row = my_list[i]
                    stock = database.Stock(i, symbol, str(row[0].split()[0]), str(row[0].split()[1]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), int(row[5]))
                    db.insert(stock)

        if index:
            db.add_indexes_to_database()

slice_list=['year'+str(year)+'month'+str(month) for year,month in list(product([1,2],list(range(1,13))))]

function = "TIME_SERIES_INTRADAY_EXTENDED"
#API gives past 2 years of data on a stock with this function
interval = "15min"
#choose slice of most recent month 
key = "RFSK33JOML91YETQ"

insert = True
index = True

##------------------------------------------------------------------------------------------

##INSERT ONE SYMBOL HERE AS FIRST ELEMENT IN LIST
symbols = ['NVDA']

#I originally wanted to use an input() to get the stock. 
#However, the dash constantly refreshes the code and it ends up re-running the input statement until the dashboard server is closed
#Also I initially set this up to be able to create databases for multiple stocks, but my my allowed API calls on the free version are extremely limited, so I went with this setup

#Make one database per year
database_paths_2022 = [symbol + "_data_2022.db" for symbol in symbols]
database_paths_2023 = [symbol + "_data_2023.db" for symbol in symbols]

slice_list_2023 = slice_list[0:5]
slice_list_2022 = slice_list[12:17]

def check_file_with_suffix(directory, suffix):
    # Get all files in the directory
    files = os.listdir(directory)

    # Check if any file has the specified suffix
    for file in files:
        if file.endswith(suffix):
            return True

    return False

directory = "C:/Users/Alex/IMGT_final"

suffix = symbols[0]+"_data_2022.db"

ds_exists = check_file_with_suffix(directory, suffix)

if ds_exists == False:
    get_data_and_make_db(symbols,database_paths_2022, slice_list_2022)
    print('getting sleepy')
    time.sleep(60) #due to API limitations, only allowed 5 calls per min. Anymore than this will break it in my experience
    print('awake now')
    get_data_and_make_db(symbols,database_paths_2023, slice_list_2023)


###Dashboard Zone

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import  html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash_bootstrap_templates import load_figure_template

#Set sidebar style
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 62.5,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        html.H2("Filters"),
        html.Hr(),
        html.P(
            "Choose a graph:", className="lead"
        ),
        dbc.Nav(
            [
                dcc.Dropdown(id = "select_viz", 
                 options = [
                     {"label":"Graph 1 (Trend)", "value":1},
                     {"label":"Graph 2 (OHLC)", "value":2},
                     {"label":"Graph 3 (Candlestick)", "value":3},
                     {"label":"Graph 4 (Volume)", "value":4}],
                     value = 1,
                ),
            html.Div(id = 'output_container', children = []),

            html.Br(),

            html.P('Choose month range:'),

            dcc.Slider(min=1, max=5, step=1, value=5, id='my-slider',tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(id='slider-output-container'),
            html.Br(),

            html.P('Change time scale:'),

            dcc.RadioItems(id ='radio2', options = 
                [{'label': 'Minutes', 'value': 1},
                {'label': 'Days', 'value': 0}]
                        , value = 0, inline=True),
            html.Div(id='radio2-output'),

            html.Br(),

            html.P('Choose display (Last two for Graph 1 only):'),

            dcc.RadioItems(id = 'radio1', options =  
            [{'label': '2023', 'value': 2023},
            {'label': '2022', 'value': 2022},
            {'label': 'Overlay', 'value': 0},
            {'label':'Both','value':1}], 
            value = 2023, inline = True),
            html.Div(id='radio1-output'),

            html.Br(),

            html.P('Graph 2 & 3 Specific Controls:'),

            dcc.RadioItems(id ='radio3', options = 
                [{'label': 'Volume', 'value': 1},
                {'label': 'No Volume', 'value': 0}]
                        , value = 0, inline=True),
            html.Div(id='radio3-output'),

            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

load_figure_template('LUX')
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

#Layout of app. Two rows
app.layout = html.Div(children = [
                dbc.Row([
                    dbc.Col(),

                    dbc.Col(html.H1("Stock Database Visualizations"),width = 8, style = {'margin-left':'10px','margin-top':'10px'})
                    ]),
                dbc.Row(
                    [dbc.Col(sidebar),
                    dbc.Col(dcc.Graph(id='main',figure = {}),width = 9, style = {'margin-left':'10px', 'margin-top':'7px', 'margin-right':'10px','padding-left':'0%'
                    })
                    ])
    ]
    
)
    
        

#Connect Graphs to Dash components
@app.callback(
    [Output(component_id='output_container', component_property = 'children'),
     Output(component_id='main', component_property = 'figure')],
    [Input(component_id='select_viz', component_property = 'value'),
     Input(component_id='my-slider', component_property = 'value'),
     Input(component_id='radio1', component_property = 'value'),
     Input(component_id='radio2', component_property = 'value'),
     Input(component_id='radio3', component_property = 'value')
     ]
)

def graph_the_things(viz_value, slider_value, radio1_value, radio2_value, radio3_value, path_2022 = database_paths_2022, path_2023 = database_paths_2023):
    print(viz_value)
    print(type(viz_value))
    
    container = 'The graph chosen by user was: {}'.format(viz_value)

    #get 2022 and 2023 data from databases 

    with sqlite3.connect(path_2022[0]) as conn:
        df_2022 = pd.read_sql_query("SELECT * FROM stock", conn)

    data_2022 = df_2022.copy()
    data_2022 = data_2022.loc[::-1].reset_index(drop=True)


    data_2022['month'] = 0

    for i in range(len(data_2022['date'])):
        data_2022.iloc[i,-1] = int(data_2022.date[i][5:7])

    with sqlite3.connect(path_2023[0]) as conn:
        df_2023 = pd.read_sql_query("SELECT * FROM stock", conn)

    data_2023 = df_2023.copy()
    data_2023 = data_2023.loc[::-1].reset_index(drop=True)


    ##create month col to enable month slider

    data_2023['month'] = 0

    for i in range(len(data_2023['date'])):
        data_2023.iloc[i,-1] = int(data_2023.date[i][5:7])

    #trim so days are equal in December
    data_2023 = data_2023[data_2023.date != '2022-12-27']
    data_2023 = data_2023[data_2023.date != '2022-12-28']

    #month_list
    month_list = [12] #starts in Dec of prev year
    temp_list = list(range(slider_value + 1))
    del temp_list[0]
    month_list = month_list + temp_list

    #change range of data based on month
    data_2023 = data_2023.loc[data_2023['month'].isin(month_list)]
    data_2022 = data_2022.loc[data_2022['month'].isin(month_list)]

    #make date+time col
    data_2023['date-time'] = 0
    data_2022['date-time'] = 0

    for i in range(len(data_2023)):
        data_2023.iloc[i,10] = data_2023.iloc[i,2] + " " + data_2023.iloc[i,3]
    
    for i in range(len(data_2022)):
        data_2022.iloc[i,10] = data_2022.iloc[i,2] + " " + data_2022.iloc[i,3]

    #Make daily data databases 

    dates = data_2023['date'].unique()
    data_by_day_2023 = pd.DataFrame()
    for i in range(len(dates)):
        rowid = i
        name = symbols[0]
        temp = data_2023[data_2023['date'] == dates[i]]
        #final_value for day
        date = temp.iloc[0,2]
        time = temp.iloc[-1,3]
        open = temp.iloc[0,4]
        high = max(temp.iloc[:,5])
        low = min(temp.iloc[:,6])
        close = temp.iloc[-1,7]
        volume = np.sum(temp.iloc[:,8])
        data_arr = np.array([rowid,name,date,time,open,high,low,close,volume])
        data_arr = np.reshape(data_arr,(1,9))
        data_df = pd.DataFrame(data_arr)
        data_df.columns = ['rowid','name','date','time','open','high','low','close','volume']
        data_by_day_2023 = pd.concat([data_by_day_2023,data_df])

    dates = data_2022['date'].unique()
    data_by_day_2022 = pd.DataFrame()
    for i in range(len(dates)):
        rowid = i
        name = symbols[0]
        temp = data_2022[data_2022['date'] == dates[i]]
        #final_value for day
        date = temp.iloc[0,2]
        time = temp.iloc[-1,3]
        open = temp.iloc[0,4]
        high = max(temp.iloc[:,5])
        low = min(temp.iloc[:,6])
        close = temp.iloc[-1,7]
        volume = np.sum(temp.iloc[:,8])
        data_arr = np.array([rowid,name,date,time,open,high,low,close,volume])
        data_arr = np.reshape(data_arr,(1,9))
        data_df = pd.DataFrame(data_arr)
        data_df.columns = ['rowid','name','date','time','open','high','low','close','volume']
        data_by_day_2022 = pd.concat([data_by_day_2022,data_df])

    #viz_value controls graph being displayed (1-4)
    if viz_value == 1:
    # radio1_value controls year/overlay components    
        if radio1_value == 2023:
    # radio2 value controls daily vs minutes
            if radio2_value == 1:
                title = "Stock trend from " + data_2023.iloc[0,2] + ' ' + data_2023.iloc[0,3] + ' to ' + data_2023.iloc[-1,2] + ' ' + data_2023.iloc[-1,3] + " ET"
                x_axis = "Date-time (Interval: " + interval + ")"
                y_axis = "Stock Price ($)"
                legend_title = "Symbol"
                fig = px.line(data_2023, y = 'close',x = 'date-time',color = data_2023['name'],
                    labels={
                    'date-time': x_axis,
                    'close': y_axis,
                    'name': legend_title
                },
                title=title)
            else:

                title = "Stock trend from " + data_2023.iloc[0,2] + ' ' + data_2023.iloc[0,3] + ' to ' + data_2023.iloc[-1,2] + ' ' + data_2023.iloc[-1,3] + " ET"
                x_axis = "Day"
                y_axis = "Stock Price ($)"
                legend_title = "Symbol"
                fig = px.line(data_by_day_2023, x = 'date', y = 'close', color = 'name',
                    labels={
                    'close': y_axis,
                    'name': legend_title
                },
                title=title)
                fig.update_layout(autotypenumbers='convert types')
        elif radio1_value == 2022:
            if radio2_value == 1:
                title = "Stock trend from " + data_2022.iloc[0,2] + ' ' + data_2022.iloc[0,3] + ' to ' + data_2022.iloc[-1,2] + ' ' + data_2022.iloc[-1,3] + " ET"
                x_axis = "Date-time (Interval: " + interval + ")"
                y_axis = "Stock Price ($)"
                legend_title = "Symbol"
                fig = px.line(data_2022, y = 'close', x = 'date-time',color = data_2022['name'],
                    labels={
                    'date-time': x_axis,
                    'close': y_axis,
                    'name': legend_title
                },
                title=title)
            else:

                title = "Stock trend from " + data_2022.iloc[0,2] + ' ' + data_2022.iloc[0,3] + ' to ' + data_2022.iloc[-1,2] + ' ' + data_2022.iloc[-1,3] + " ET"
                x_axis = "Day"
                y_axis = "Stock Price ($)"
                legend_title = "Symbol"
                fig = px.line(data_by_day_2022, x = 'date', y = 'close', color = 'name',
                    labels={

                    'close': y_axis,
                    'name': legend_title
                },
                title=title)
                fig.update_layout(autotypenumbers='convert types')
        elif radio1_value == 0:

            if radio2_value == 0:

                title1 = symbols[0] + " stock trend from " + data_2022.iloc[0,2] + ' to ' + data_2022.iloc[-1,2] 
                title2 = symbols[0] + " stock trend from " + data_2023.iloc[0,2] + ' to ' + data_2023.iloc[-1,2] 
                y_axis = "Stock Price ($)"
                fig = make_subplots(rows=2, cols=1, shared_xaxes=False, 
                        vertical_spacing=0.03, subplot_titles=(title1, title2), 
                        row_width=[0.5, 0.5])

                fig.append_trace(go.Scatter(x = data_by_day_2022['date'],y = data_by_day_2022['close'],
                    mode = 'lines', name = '2022'), row=1, col=1)
                fig.update_yaxes(title_text ='Stock Price ($)',row = 1, col = 1)
                fig.update_layout(autotypenumbers='convert types')
                fig.append_trace(go.Scatter(x = data_by_day_2023['date'],y = data_by_day_2023['close'],mode = 'lines', name = '2023'), row=2, col=1)
                fig.update_layout(autotypenumbers='convert types')
                fig.update_yaxes(title_text ='Stock Price ($)',row = 2, col = 1)

            else:
                title1 = symbols[0] + " stock trend from " + data_2022.iloc[0,2] + ' to ' + data_2022.iloc[-1,2] 
                title2 = symbols[0] + " stock trend from " + data_2023.iloc[0,2] + ' to ' + data_2023.iloc[-1,2] 
                y_axis = "Stock Price ($)"
                x_axis = "Date-time (Interval: " + interval + ")"
                fig = make_subplots(rows=2, cols=1, shared_xaxes=False, 
                        vertical_spacing=0.03, subplot_titles=(title1, title2), 
                        row_width=[0.5, 0.5])

                fig.append_trace(go.Scatter(x = data_2022['date-time'],y = data_2022['close'],
                    mode = 'lines', name = '2022'), row=1, col=1)
                fig.update_yaxes(title_text ='Stock Price ($)',row = 1, col = 1)
                fig.update_layout(autotypenumbers='convert types')
                fig.append_trace(go.Scatter(x = data_2023['date-time'],y = data_2023['close'],mode = 'lines', name = '2023'), row=2, col=1)
                fig.update_layout(autotypenumbers='convert types')
                fig.update_yaxes(title_text ='Stock Price ($)',row = 2, col = 1)
                fig.update_xaxes(title_text = x_axis,row = 2, col = 1)
        else:
            #2022
            if radio2_value == 0:
                title1 = " stock trend from " + data_2022.iloc[0,2] + ' to ' + data_2022.iloc[-1,2] 
                title2 = " and trend line from " + data_2023.iloc[0,2] + ' to ' + data_2023.iloc[-1,2] 
                title = symbols[0] + title1 + title2
                x_axis = "Day"
                y_axis = "Stock Price ($)"
                fig = px.line(data_by_day_2022,x = 'date',y = 'close', color_discrete_sequence=['purple'],
                    labels={
                    'close': y_axis
                }, title = title)
                fig.update_layout(autotypenumbers='convert types')

                #2023

                fig2 = px.line(data_by_day_2023, x = 'date',y = 'close', color_discrete_sequence=['orange'])
                fig2.update_layout(autotypenumbers='convert types')
                fig.add_traces(list(fig2.select_traces()))
                name = ['2022','2023']

                for i in range(len(fig.data)):
                    fig.data[i]['name'] = name[i]
                    fig.data[i]['showlegend'] = True

            else:
                title1 = " stock trend from " + data_2022.iloc[0,2] + ' to ' + data_2022.iloc[-1,2] 
                title2 = " and trend line from " + data_2023.iloc[0,2] + ' to ' + data_2023.iloc[-1,2] 
                title = symbols[0] + title1 + title2
                x_axis = "Date-time (Interval: " + interval + ")"
                y_axis = "Stock Price ($)"
                fig = px.line(data_2022,x = 'date-time',y = 'close', color_discrete_sequence=['purple'],
                    labels={
                    'date-time': x_axis,
                    'close': y_axis
                },
                title=title)
                fig.update_layout(autotypenumbers='convert types')

                #2023

                fig2 = px.line(data_2023, x = 'date-time',y = 'close', color_discrete_sequence=['orange'])
                fig2.update_layout(autotypenumbers='convert types')
                fig.add_traces(list(fig2.select_traces()))
                name = ['2022','2023']

                for i in range(len(fig.data)):
                    fig.data[i]['name'] = name[i]
                    fig.data[i]['showlegend'] = True

        

    if viz_value == 2:

        if radio1_value == 2023:
            if radio2_value == 0:
                if radio3_value == 0:
                #radio3 value is the volume toggle

                    fig = go.Figure(data=go.Ohlc(x=data_by_day_2023['date'],
                                open=data_by_day_2023['open'],
                                high=data_by_day_2023['high'],
                                low=data_by_day_2023['low'],
                                close=data_by_day_2023['close']))
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2023.iloc[0,2] + ' ' + data_2023.iloc[0,3] + ' to ' + data_2023.iloc[-1,2] + ' ' + data_2023.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)', xaxis_title = 'date')
                    # fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),paper_bgcolor="LightSteelBlue",)
                    # fig.update_layout(width=int(width))
                else: 
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.06, subplot_titles=('OHLC', 'Volume'), 
                    row_width=[0.2, 0.7])

                    # Plot OHLC on 1st row
                    fig.add_trace(go.Ohlc(x=data_by_day_2023["date"], open=data_by_day_2023["open"], high=data_by_day_2023["high"],
                                    low=data_by_day_2023["low"], close=data_by_day_2023["close"], name="OHLC"), 
                                    row=1, col=1)
                    # Bar trace for volumes on 2nd row without legend
                    fig.add_trace(go.Bar(x=data_by_day_2023['date'], y=data_by_day_2023['volume'], showlegend=False), row=2, col=1)

                    # Do not show OHLC's rangeslider plot 
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2023.iloc[0,2] + ' ' + data_2023.iloc[0,3] + ' to ' + data_2023.iloc[-1,2] + ' ' + data_2023.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)')

            else: 
                if radio3_value == 0:
                    fig = go.Figure(data=go.Ohlc(x=data_2023['date-time'],
                                open=data_2023['open'],
                                high=data_2023['high'],
                                low=data_2023['low'],
                                close=data_2023['close']))
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2023.iloc[0,2] + ' ' + data_2023.iloc[0,3] + ' to ' + data_2023.iloc[-1,2] + ' ' + data_2023.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)', xaxis_title = 'date')
                else:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.06, subplot_titles=('OHLC', 'Volume'), 
                    row_width=[0.2, 0.7])

                    # Plot OHLC on 1st row
                    fig.add_trace(go.Ohlc(x=data_2023['date-time'], open=data_2023["open"], high=data_2023["high"],
                                    low=data_2023["low"], close=data_2023["close"], name="OHLC"), 
                                    row=1, col=1)
                    # Bar trace for volumes on 2nd row without legend
                    fig.add_trace(go.Bar(x=data_2023['date-time'], y=data_2023['volume'], showlegend=False), row=2, col=1)

                    # Do not show OHLC's rangeslider plot 
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2023.iloc[0,2] + ' ' + data_2023.iloc[0,3] + ' to ' + data_2023.iloc[-1,2] + ' ' + data_2023.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)')
        else:
            if radio2_value == 0:
                if radio3_value == 0:
                    fig = go.Figure(data=go.Ohlc(x=data_by_day_2022['date'],
                                open=data_by_day_2022['open'],
                                high=data_by_day_2022['high'],
                                low=data_by_day_2022['low'],
                                close=data_by_day_2022['close']))
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2022.iloc[0,2] + ' ' + data_2022.iloc[0,3] + ' to ' + data_2022.iloc[-1,2] + ' ' + data_2022.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)', xaxis_title = 'date')
                    # fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),paper_bgcolor="LightSteelBlue",)
                    # fig.update_layout(width=int(width))
                else: 
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.06, subplot_titles=('OHLC', 'Volume'), 
                    row_width=[0.2, 0.7])

                    # Plot OHLC on 1st row
                    fig.add_trace(go.Ohlc(x=data_by_day_2022["date"], open=data_by_day_2022["open"], high=data_by_day_2022["high"],
                                    low=data_by_day_2022["low"], close=data_by_day_2022["close"], name="OHLC"), 
                                    row=1, col=1)
                    # Bar trace for volumes on 2nd row without legend
                    fig.add_trace(go.Bar(x=data_by_day_2022['date'], y=data_by_day_2022['volume'], showlegend=False), row=2, col=1)

                    # Do not show OHLC's rangeslider plot 
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2022.iloc[0,2] + ' ' + data_2022.iloc[0,3] + ' to ' + data_2022.iloc[-1,2] + ' ' + data_2022.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)')

            else: 
                if radio3_value == 0:
                    fig = go.Figure(data=go.Ohlc(x=data_2022['date-time'],
                                open=data_2022['open'],
                                high=data_2022['high'],
                                low=data_2022['low'],
                                close=data_2022['close']))
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')

                    title = "Stock trend from " + data_2022.iloc[0,2] + ' ' + data_2022.iloc[0,3] + ' to ' + data_2022.iloc[-1,2] + ' ' + data_2022.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)', xaxis_title = 'date')
                else:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.06, subplot_titles=('OHLC', 'Volume'), 
                    row_width=[0.2, 0.7])

                    # Plot OHLC on 1st row
                    fig.add_trace(go.Ohlc(x=data_2022['date-time'], open=data_2022["open"], high=data_2022["high"],
                                    low=data_2022["low"], close=data_2022["close"], name="OHLC"), 
                                    row=1, col=1)
                    # Bar trace for volumes on 2nd row without legend
                    fig.add_trace(go.Bar(x=data_2022['date-time'], y=data_2022['volume'], showlegend=False), row=2, col=1)

                    # Do not show OHLC's rangeslider plot 
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')

                    title = "Stock trend from " + data_2022.iloc[0,2] + ' ' + data_2022.iloc[0,3] + ' to ' + data_2022.iloc[-1,2] + ' ' + data_2022.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)')
        
    if viz_value == 3:
        if radio1_value == 2023:
            if radio2_value == 0:
                if radio3_value == 0:
                    fig = go.Figure(data=go.Candlestick(x=data_by_day_2023['date'],
                                open=data_by_day_2023['open'],
                                high=data_by_day_2023['high'],
                                low=data_by_day_2023['low'],
                                close=data_by_day_2023['close']))
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2023.iloc[0,2] + ' ' + data_2023.iloc[0,3] + ' to ' + data_2023.iloc[-1,2] + ' ' + data_2023.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)', xaxis_title = 'date')
                    # fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),paper_bgcolor="LightSteelBlue",)
                    # fig.update_layout(width=int(width))
                else: 
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.06, subplot_titles=('Candlestick', 'Volume'), 
                    row_width=[0.2, 0.7])

                    
                    fig.add_trace(go.Candlestick(x=data_by_day_2023["date"], open=data_by_day_2023["open"], high=data_by_day_2023["high"],
                                    low=data_by_day_2023["low"], close=data_by_day_2023["close"], name='Candlestick'), 
                                    row=1, col=1)
                    # Bar trace for volumes on 2nd row without legend
                    fig.add_trace(go.Bar(x=data_by_day_2023['date'], y=data_by_day_2023['volume'], showlegend=False), row=2, col=1)

                    
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2023.iloc[0,2] + ' ' + data_2023.iloc[0,3] + ' to ' + data_2023.iloc[-1,2] + ' ' + data_2023.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)')

            else: 
                if radio3_value == 0:
                    fig = go.Figure(data=go.Candlestick(x=data_2023['date-time'],
                                open=data_2023['open'],
                                high=data_2023['high'],
                                low=data_2023['low'],
                                close=data_2023['close']))
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2023.iloc[0,2] + ' ' + data_2023.iloc[0,3] + ' to ' + data_2023.iloc[-1,2] + ' ' + data_2023.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)', xaxis_title = 'date')
                else:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.06, subplot_titles=('Candlestick', 'Volume'), 
                    row_width=[0.2, 0.7])

                    
                    fig.add_trace(go.Candlestick(x=data_2023['date-time'], open=data_2023["open"], high=data_2023["high"],
                                    low=data_2023["low"], close=data_2023["close"], name='Candlestick'), 
                                    row=1, col=1)
                    # Bar trace for volumes on 2nd row without legend
                    fig.add_trace(go.Bar(x=data_2023['date-time'], y=data_2023['volume'], showlegend=False), row=2, col=1)

                     
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2023.iloc[0,2] + ' ' + data_2023.iloc[0,3] + ' to ' + data_2023.iloc[-1,2] + ' ' + data_2023.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)')
        else:
            if radio2_value == 0:
                if radio3_value == 0:
                    fig = go.Figure(data=go.Candlestick(x=data_by_day_2022['date'],
                                open=data_by_day_2022['open'],
                                high=data_by_day_2022['high'],
                                low=data_by_day_2022['low'],
                                close=data_by_day_2022['close']))
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    # fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),paper_bgcolor="LightSteelBlue",)
                    # fig.update_layout(width=int(width))
                    title = "Stock trend from " + data_2022.iloc[0,2] + ' ' + data_2022.iloc[0,3] + ' to ' + data_2022.iloc[-1,2] + ' ' + data_2022.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)', xaxis_title = 'date')
                else: 
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.06, subplot_titles=('Candlestick', 'Volume'), 
                    row_width=[0.2, 0.7])

                
                    fig.add_trace(go.Candlestick(x=data_by_day_2022["date"], open=data_by_day_2022["open"], high=data_by_day_2022["high"],
                                    low=data_by_day_2022["low"], close=data_by_day_2022["close"], name='Candlestick'), 
                                    row=1, col=1)
                    # Bar trace for volumes on 2nd row without legend
                    fig.add_trace(go.Bar(x=data_by_day_2022['date'], y=data_by_day_2022['volume'], showlegend=False), row=2, col=1)

                   
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2022.iloc[0,2] + ' ' + data_2022.iloc[0,3] + ' to ' + data_2022.iloc[-1,2] + ' ' + data_2022.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)')

            else: 
                if radio3_value == 0:
                    fig = go.Figure(data=go.Candlestick(x=data_2022['date-time'],
                                open=data_2022['open'],
                                high=data_2022['high'],
                                low=data_2022['low'],
                                close=data_2022['close']))
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2022.iloc[0,2] + ' ' + data_2022.iloc[0,3] + ' to ' + data_2022.iloc[-1,2] + ' ' + data_2022.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)', xaxis_title = 'date')
                else:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.06, subplot_titles=('Candlestick', 'Volume'), 
                    row_width=[0.2, 0.7])

                    
                    fig.add_trace(go.Candlestick(x=data_2022['date-time'], open=data_2022["open"], high=data_2022["high"],
                                    low=data_2022["low"], close=data_2022["close"], name='Candlestick'), 
                                    row=1, col=1)
                    # Bar trace for volumes on 2nd row without legend
                    fig.add_trace(go.Bar(x=data_2022['date-time'], y=data_2022['volume'], showlegend=False), row=2, col=1)

                    
                    fig.update(layout_xaxis_rangeslider_visible=False)
                    fig.update_layout(autotypenumbers='convert types')
                    title = "Stock trend from " + data_2022.iloc[0,2] + ' ' + data_2022.iloc[0,3] + ' to ' + data_2022.iloc[-1,2] + ' ' + data_2022.iloc[-1,3] + " ET"
                    fig.update_layout(title_text= title, yaxis_title = 'Price ($)', xaxis_title = 'date')
    if viz_value == 4:
        if radio1_value == 2023:
            if radio2_value == 0:
                title = symbols[0] + " volume from " + data_2023.iloc[0,2] + ' to ' + data_2023.iloc[-1,2] 
                fig = px.bar(data_by_day_2023, x='date', y='volume',title = title)
                fig.update_layout(autotypenumbers='convert types')

            else: 
                title = symbols[0] + " volume from " + data_2023.iloc[0,2] + ' to ' + data_2023.iloc[-1,2] 
                fig = px.bar(data_2023, x='date-time', y='volume',title = title,
                             labels = {'date-time':'date'})
        else:
            if radio2_value == 0:
                title = symbols[0] + " volume from " + data_2022.iloc[0,2] + ' to ' + data_2023.iloc[-1,2] 
                fig = px.bar(data_by_day_2022, x='date', y='volume',title = title)
                fig.update_layout(autotypenumbers='convert types')

            else: 
                title = symbols[0] + " volume from " + data_2022.iloc[0,2] + ' to ' + data_2022.iloc[-1,2] 
                fig = px.bar(data_2022, x='date-time', y='volume',title = title,
                             labels = {'date-time':'date'})
    return container, fig
        


if __name__ =='__main__':
    app.run_server(debug=True)