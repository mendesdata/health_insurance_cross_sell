# Import Libraries

import pandas               as pd
import numpy                as np
import streamlit            as st
import plotly.express       as px
import plotly.graph_objects as go
import matplotlib.pyplot    as plt
import scikitplot           as skplt

import requests
import json
import math
import datetime
import os
import inflection

from  api.HICS import HICS


# rename columns
def rename_columns( df ):
    title = lambda x: inflection.titleize( x )
    snakecase = lambda x: inflection.underscore( x )
    spaces = lambda x: x.replace(" ", "")

    cols_old = list( df.columns )
    cols_old = list( map( title, cols_old ) )
    cols_old = list( map( spaces, cols_old ) )
    cols_new = list( map( snakecase, cols_old ) )
    
    df.columns = cols_new

    return df


def load_data( file ):
    # colect original test dataset
    df = pd.read_csv( file )
    
    return df

def apply_model( x_test, local=True ):
    # drop sales columns
    #y_test = x_test['response'].values
    #x_test = x_test.drop( ['response'], axis=1 )

    # convert dataframe to json
    data_json = json.dumps( x_test.to_dict( orient='records' ) )

    # Render´s Server request
    url = 'https://hics-ws.onrender.com/hics/predict'

    # local request
    if local:
        url = 'http://0.0.0.0:5000/hics/predict'
        
    # API CALL  
    header = { 'Content-type' : 'application/json' }
    response = requests.post( url, data=data_json, headers=header ) 

    # return dataframe with predictions
    df = pd.DataFrame( response.json(), columns=response.json()[0].keys() )
    #df = df[['id', 'score', 'response']]
    df = df.sort_values( 'score', ascending=False ).reset_index( drop=True)  

    return df

def data_metrics( df ):
    customers          = df['id'].count()
    neg_customers      = df[ df['response'] == 0]['id'].count()
    neg_customers_perc = round( neg_customers / customers * 100, 2)
    pos_customers      = df[ df['response'] == 1]['id'].count()
    pos_customers_perc = round( 100 - neg_customers_perc, 2)

    with st.container():
        col1, col2, col3 = st.columns ( 3, gap='small' )
    
        with col1:
            st.metric( label='Customers',  
                       value='{:,.0f}'.format( customers ), 
                       help='Number of customers' )      

        with col2:
            st.metric( label='Uninterested Customers',  
                       value='{:,.0f}'.format( neg_customers ), 
                       delta='{:,.2f}'.format( neg_customers_perc )+'%', 
                       help='Number of uninterested customers' )      

        with col3:
            st.metric( label='Interested Customers',  
                       value='{:,.0f}'.format( pos_customers ), 
                       delta='{:,.2f}'.format( pos_customers_perc )+'%',  
                       help='Number of interested customers')                   

        return None
    
def performance_curves( df ):
    with st.container():
        y = df['response']
        yhat = []

        for i in range(0, df.shape[0] ):
            aux = [ df.loc[i, 'negative_score'], df.loc[i, 'score'] ]
            yhat.append( aux )

        st.markdown('#### Cumulative Gains and Lift Curves')

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # plot curves
        skplt.metrics.plot_cumulative_gain( y, yhat, ax=axes[0], title='Cumulative Gains Curve' )
        skplt.metrics.plot_lift_curve( y, yhat, ax=axes[1], title='Lift Curve' )

        st.plotly_chart( fig, use_container_width=True )

    return None
    
def ranking_data( df, cost_per_customer, average_ticket ):
    df['ranking']        = df.index+1
    df['percentage']     = round( df['ranking'] / df['id'].count(), 4 )
    df['precision']      = round( df['response'].cumsum() / df['ranking'], 4 )

    df['recall']         = round( df['response'].cumsum() / df['response'].sum(), 4 )
    df['recall_cost']    = round( df['ranking'] * cost_per_customer, 2)
    df['recall_revenue'] = round( df['response'].sum() * df['recall'] * average_ticket, 2)

    df['random_cost']    = round( df['ranking'] * cost_per_customer, 2 ) 
    df['random_revenue'] = round( df['ranking'] / df['id'].count() * df['response'].sum() * average_ticket, 2 ) 

    return df


def cost_forecast_chart( df ):
    with st.container():
        _perc        = []
        _cost_model  = []
        _cost_random = []

        for i in range(1, 11):
            perc = i / 10

            # cost model
            aux = df[ df['recall'] >= perc].sort_values( 'ranking' ).reset_index()
            _perc.append( i * 10 )
            _cost_model.append( aux.loc[0, 'recall_cost'] )

            # cost random
            aux = df[ df['percentage'] >= perc].sort_values( 'ranking' ).reset_index()
            _cost_random.append( aux.loc[0, 'random_cost'] )

        fig = go.Figure(data=[
            go.Bar(name='Hics Model'  , x=_perc, y=_cost_model,  textposition='auto', text=_cost_model ),
            go.Bar(name='Random Model', x=_perc, y=_cost_random, textposition='auto', text=_cost_random ) 
            ]  )
        
        fig.update_layout( title='Cost Forecast Chart', xaxis_title='Interested Customers (%)', yaxis_title='Operational Cost (U$)' )
        st.plotly_chart( fig, use_container_width=True )

    return None

def revenue_forecast_chart( df ):
    with st.container():
        _perc           = []
        _revenue_model  = []
        _revenue_random = []

        for i in range(1, 11):
            perc = i / 10
            aux = df[ df['percentage'] >= perc].sort_values( 'ranking').reset_index()

            _perc.append( i * 10 )
            _revenue_model.append( aux.loc[0, 'recall_revenue'] )
            _revenue_random.append( aux.loc[0, 'random_revenue'] )


        fig = go.Figure(data=[
            go.Bar(name='Hics Model'  , x=_perc, y=_revenue_model,  textposition='auto', text=_revenue_model ),
            go.Bar(name='Random Model', x=_perc, y=_revenue_random, textposition='auto', text=_revenue_random ) 
            ]  )

        fig.update_layout(title='Revenue Forecast Chart', xaxis_title='Customers (%)', yaxis_title='Revenue (U$)' )
        st.plotly_chart( fig, use_container_width=True )
        
    return None

def customers_list( df, perc ):
    percentage = perc / 100
    aux = df[ df['recall'] <= percentage].reset_index()
    max_rank = aux['ranking'].max()

    rank    = aux.loc[max_rank-1, 'ranking']
    cost    = aux.loc[max_rank-1, 'recall_cost']
    revenue = aux.loc[max_rank-1, 'recall_revenue']

    with st.container():
        col1, col2, col3, col4, col5 = st.columns ( 5, gap='small' )

        with col1: 
            # interested customers metric
            value = int( percentage * df['response'].sum() )
            st.metric( label='Interested Customers',  
                       value='{:,.0f}'.format( value ), 
                       delta='{:,.0f}'.format( perc )+'%', 
                       help='Number of interested customers' )  
        with col2:
            # customers contacted metric
            delta = round( rank / df['id'].count() * 100, 2)
            st.metric( label='Customers Contacted',  
                       value='{:,.0f}'.format( rank ), 
                       delta='{:,.2f}'.format( delta )+'%', 
                       help='Number of customers contacted' )  
        with col3:
            # cost forecast metric
            st.metric( label='Cost Forecast',  
                       value='{:,.2f}'.format( cost ), 
                       help='Amount of Predicted Cost' )  
        with col4:
            # revenue forecast metric
            st.metric( label='Revenue Forecast',  
                       value='{:,.2f}'.format( revenue ), 
                       help='Amount of Predicted Revenue' )  

        with col5: 
            st.markdown('Customers List')
            #aux = aux[['ranking', 'id', 'age', 'gender', 'region_code']].rename( columns = {'id' : 'customer_id'} )            
            #st.dataframe( aux, hide_index=True )

    return None

def customer_profile( df ):
    with st.container():
        col1, col2, col3 = st.columns( 3, gap='small' )

        # old_age column
        median_age = df['age'].median()
        df['old_age'] = df['age'].apply( lambda x : 'over 36y' if x > median_age else 'until 36y' )   

        df_pos_response = df[ df['response'] == 1]
        df['response'] = 1

        with col1:
            #fig =go.Figure(go.Sunburst(df, path=['gender', 'old_age', 'vehicle_age'], values='response', textinfo='label+percent' ) )
            fig = px.sunburst(df, path=['gender', 'old_age', 'vehicle_age'], values='response')
            fig.update_traces(textinfo="label+percent root")
            fig.update_layout(title='All Customers' )
            st.plotly_chart( fig, use_container_width=True )

        with col2:
            fig = px.sunburst(df_pos_response, path=['gender', 'old_age', 'vehicle_age'], values='response')
            fig.update_traces(textinfo="label+percent root")
            fig.update_layout(title='Interested Customers' )
            st.plotly_chart( fig, use_container_width=True )

        with col3:
            st.markdown('## Insights List')
            st.markdown('**1.** **74% dos clientes interessados** no seguro automóvel possuem veículos com idade entre 1 e 2 anos. Esse grupo representa **54% do total de clientes**')
            st.markdown('**2.** **45% dos clientes interessados** no seguro automóvel são do gênero masculino com idade acima dos 36 anos. Esse grupo representa **31% do total de clientes**')
            st.markdown('**3.** Apenas **29% dos clientes interessados** no seguro automóvel possuem idade até os 36 anos. Este grupo representa **50% do total de clientes**')

    return None

# >>> Main Function
def main():
    # configure page title
    st.set_page_config( page_title='Health Insurance Cross Sell Monitor - Main Page', layout='wide' )  

    # create test dataset
    x_test = load_data( 'data/test.csv' )
    x_test = rename_columns( x_test )

    st.title( 'Welcome to H.I.C.S Monitor' )
    st.markdown('### Health Insurance Cross Sell' )

    # sidebar area
    st.sidebar.markdown('# Filters')

    cost_per_customer = st.sidebar.number_input('Cost per contact')
    average_ticket    = st.sidebar.number_input('Average Ticket')

    st.sidebar.markdown('# Customers List')
    score_slider      = st.sidebar.slider('Target of interested customers (%):', min_value=1, max_value=100 )
   
    button = st.sidebar.button('Apply Model', type='primary')

    if button:
        df_score = apply_model( x_test )

        data_metrics( df_score )
        df_rank  = ranking_data( df_score, cost_per_customer, average_ticket )
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs( [ 'Ranking Curves',
                                                  'Cost Forecast', 
                                                  'Revenue Forecast', 
                                                  'Customers List',
                                                  'Insights: Customer Profile'] )

        with tab1: 
            performance_curves( df_score )

        with tab2: 
            cost_forecast_chart( df_rank )   
            customers_list(df_rank, score_slider)

        with tab3:
            revenue_forecast_chart( df_rank )            

        with tab4: 
            customers_list(df_rank, score_slider)

        with tab5:
            customer_profile( df_score )

    return None
   
# >>> Call Main Function
if __name__ == "__main__":
    main()
    