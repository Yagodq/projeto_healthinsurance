import pickle
import pandas as pd
import os
from flask import Flask, request, Response
from healthinsurance.HealthInsurance import HealthInsurance

# loading model
#path = 'C:\\Users\\yago2\\OneDrive\\Documentos\\Repus\\healt_insurance_cross_sell_project\\'
model = pickle.load( open ( 'models/model_health_insurance.pkl', 'rb') )

# initialize API
app = Flask( __name__ )

@app.route( '/healthinsurance/predict', methods=['POST'] )

def health_insurance_predict():
    test_json = request.get_json()

    if test_json: # there is data

        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )

        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instantiate HealtInsurance class
        pipeline = HealthInsurance()
        # data cleaning
        df1 = pipeline.rename_columns( test_raw )
        # feature engineering
        df2 = pipeline.feature_engineering( df1 )
        # data encoding
        df3 = pipeline.data_encoding( df2 )
        # data rescaling
        df4 = pipeline.data_rescaling( df3 )
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df4 )

        return df_response
    
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
#if __name__ == '__main__':
#   app.run( '0.0.0.0', debug=True )

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run( '0.0.0.0', port=port )