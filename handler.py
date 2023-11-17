from   flask    import Flask, request, Response
from   api.HICS import HICS
import pandas   as pd
import pickle
import os

# loading model 
path_model = '/home/datamendes/comunidadeds/projetos/health_insurance_cross_sell/models/'
#path_model = os.environ.get( 'path_model', '/home/datamendes/comunidadeds/projetos/health_insurance_cross_sell/models/')
model      = pickle.load( open( path_model + 'hics_model.pkl', 'rb' ) )

# Initialize API
app = Flask( __name__ )
@app.route( '/hics/predict', methods=['POST'] )
def hics_predict():
    test_json = request.get_json()

    # there is data
    if test_json:

        # Unique Example
        if isinstance( test_json, dict ):
            df_test_raw = pd.DataFrame( test_json, index=[0] )
        
        # Multiple Examples
        else:
            df_test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )

        df_raw = df_test_raw.copy()

        # Instanciate HICS Class
        pipeline = HICS()

        # data clening
        df = pipeline.data_cleaning( df_raw )

        # feature enginering
        df = pipeline.feature_engineering( df )

        # data preparation
        df = pipeline.data_preparation( df )

        # features selection
        df = pipeline.feature_selection( df )

        # Prediction
        df_response = pipeline.get_prediction( model, df_test_raw, df )

        return df_response

    # there is no data
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    

if __name__ == '__main__':
    app.run( '0.0.0.0' )