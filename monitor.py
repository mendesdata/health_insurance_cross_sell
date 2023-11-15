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

from  api.HICS import HICS


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
        #fig = px.bar( df, x='percentage', y='revenue_model', text='revenue_model') 
        #fig.update_layout(title='Revenue Forecast Chart', xaxis_title='Interested Customers (%)', yaxis_title='Revenue (U$)' )
        #st.plotly_chart( fig, use_container_width=True )

        y = df['response']
        yhat = []

        for i in range(0, df.shape[0] ):
            aux = [ df.loc[i, 'negative_score'], df.loc[i, 'score'] ]
            yhat.append( aux )

        st.markdown('### Ranking Curves')

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # plot curves
        skplt.metrics.plot_cumulative_gain( y, yhat, ax=axes[0], title='Cumulative Gains Curve' )
        skplt.metrics.plot_lift_curve( y, yhat, ax=axes[1], title='Lift Curve' )

        st.plotly_chart( fig, use_container_width=True )
    
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

def customers_list( df, recall_at_k ):
    aux = df[ df['recall'] <= recall_at_k].reset_index()

    max_rank = aux['ranking'].max()

    rank    = aux.loc[max_rank-1, 'ranking']
    cost    = aux.loc[max_rank-1, 'cost']
    revenue = aux.loc[max_rank-1, 'revenue']

    aux = aux[['ranking', 'id', 'age', 'gender', 'region_code']].rename( columns = {'id' : 'customer_id'} )

    return aux, rank, cost, revenue

# >>> Main Function
def main():
    # configure page title
    st.set_page_config( page_title='Health Insurance Cross Sell Monitor - Main Page', layout='wide' )  

    # create test dataset
    x_test = load_data( 'data/test.csv' )

    st.title( 'Welcome to H.I.C.S Monitor' )
    #st.markdown( subtitle )

    # sidebar area
    st.sidebar.markdown('# Filters')

    cost_per_customer = st.sidebar.number_input('Cost per contact')
    average_ticket    = st.sidebar.number_input('Average Ticket')
    score_slider      = st.sidebar.slider('Target of interested customers', min_value=1, max_value=100 )
   
    button = st.sidebar.button('Apply Model', type='primary')

    if button:
        df_score = apply_model( x_test )
        data_metrics( df_score )
        df_rank  = ranking_data( df_score, cost_per_customer, average_ticket )

        st.dataframe( df_rank )
        
        tab1, tab2, tab3, tab4 = st.tabs( [ 'Ranking Curves',
                                            'Cost Forecast', 
                                            'Revenue Forecast', 
                                            'Proposed Goal'] )

        with tab1: 
            performance_curves( df_score )

        with tab2: 
            cost_forecast_chart( df_rank )   

        with tab3:
            revenue_forecast_chart( df_rank )            

        with tab4: 
            df_customers, rank, cost, revenue  = customers_list( df_rank, score_slider/100 )

            with st.container():
                col1, col2 = st.columns ( 2, gap='small' )

                with col1: 
                    delta = round( rank / df_score['id'].count() * 100, 2)
                    st.metric( label='Customers Contacted',  
                               value='{:,.0f}'.format( rank ), 
                               delta='{:,.2f}'.format( delta )+'%', 
                               help='Number of customers contacted' )  
                    
                #with col2: 
                    value = int( score_slider * 0.01 * df_score['response'].sum() )
                    st.metric( label='Interested Customers',  
                               value='{:,.0f}'.format( value ), 
                               delta='{:,.0f}'.format( score_slider )+'%', 
                               help='Number of interested customers' )  

                #with col3: 
                    st.metric( label='Cost Forecast',  
                               value='{:,.2f}'.format( cost ), 
                               help='Amount of Predicted Cost' )  
                    
                #with col4:
                    st.metric( label='Revenue Forecast',  
                               value='{:,.2f}'.format( revenue ), 
                               help='Amount of Predicted Revenue' )  

                with col2: 
                    st.markdown('Customers List')
                    st.dataframe( df_customers, hide_index=True )


    return None
   
# >>> Call Main Function
if __name__ == "__main__":
    main()
    