import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime
import os

class HICS( object ):
    def __init__(self):
        self.path_params                 = os.environ.get( 'path_params', '/home/datamendes/comunidadeds/projetos/health_insurance_cross_sell/parameters/')
        self.annual_premium_scaler       = pickle.load( open( self.path_params + 'annual_premium_scaler.pkl', 'rb' ) )
        self.age_scaler                  = pickle.load( open( self.path_params + 'age_scaler.pkl', 'rb' ) )
        self.vintage_scaler              = pickle.load( open( self.path_params + 'vintage_scaler.pkl', 'rb' ) )
        self.region_code_scaler          = pickle.load( open( self.path_params + 'region_code_scaler.pkl', 'rb' ) )
        self.policy_sales_channel_scaler = pickle.load( open( self.path_params + 'policy_sales_channel_scaler.pkl', 'rb' ) )

    def data_cleaning( self, df ):
        title = lambda x: inflection.titleize( x )
        snakecase = lambda x: inflection.underscore( x )
        spaces = lambda x: x.replace(" ", "")

        cols_old = list( df.columns )
        cols_old = list( map( title, cols_old ) )
        cols_old = list( map( spaces, cols_old ) )
        cols_new = list( map( snakecase, cols_old ) )

        #cols_new = ['id','gender', 'age', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age', 
        #             'vehicle_damage', 'annual_premium', 'policy_sales_channel', 'vintage']
        
        df.columns = cols_new

        df['region_code']          = df['region_code'].astype( int )
        df['policy_sales_channel'] = df['policy_sales_channel'].astype( int )

        return df

    def feature_engineering( self, df ):
        # male column
        df['male'] = df['gender'].apply( lambda x : 1 if x == 'Male' else 0 )

        # old_age column
        median_age = df['age'].median()
        df['old_age'] = df['age'].apply( lambda x : 1 if x > median_age else 0 )    

        # old_annual_premium column
        median_premium = df['annual_premium'].median()
        df['old_annual_premium'] = df['annual_premium'].apply( lambda x : 1 if x > median_premium else 0 )

        # old_vintage column
        median_premium = df['vintage'].median()
        df['old_vintage'] = df['vintage'].apply( lambda x : 1 if x > median_premium else 0 )
        
        return df

    def data_preparation( self, df ):
        df['annual_premium']       = self.annual_premium_scaler.transform( df[['annual_premium']].values )
        df['age']                  = self.age_scaler.transform( df[['age']].values )    
        df['vintage']              = self.vintage_scaler.transform( df[['vintage']].values )            
        df['region_code']          = self.region_code_scaler.transform( df[['region_code']].values )
        df['policy_sales_channel'] = self.policy_sales_channel_scaler.transform( df[['policy_sales_channel']].values )

        # Enconding     
        bool_dict = { 'Yes' : 1, 'No' : 0 }
        df['vehicle_damage'] = df['vehicle_damage'].map( bool_dict )
    
        # Apply Ordinal Encoding - vehicle_age
        _dict = { '< 1 Year' : 1, '1-2 Year' : 2, '> 2 Years' : 3 }
        df['vehicle_age'] = df['vehicle_age'].map( _dict )

        # drop unnecessary columns
        cols_drop = ['gender', 'old_age', 'old_annual_premium', 'old_vintage']
        df = df.drop( cols_drop, axis=1 )
            
        return df
        
    
    def feature_selection( self, df ):
        final_features = ['vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage', 'policy_sales_channel', 'previously_insured']
        #final_features.extend( ['id'] )

        return df[final_features]
    
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        score_prediction = model.predict_proba( test_data )

        # join pred into original data
        original_data['score'] = score_prediction[:,1].tolist()

        return original_data.to_json( orient='records', date_format='iso' )
