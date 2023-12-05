import pickle
import numpy as np
import pandas as pd
import inflection

class HealthInsurance:
    
    def __init__(self):
        self.home_path              = 'C:\\Users\\yago2\\OneDrive\\Documentos\\Repus\\healt_insurance_cross_sell_project\\'
        self.annual_premium_scaler  = pickle.load( open ( self.home_path + 'src/features/annual_premium_scaler.pkl', 'rb'))
        self.age_scaler             = pickle.load( open ( self.home_path + 'src/features/age_scaler.pkl', 'rb' ))
        self.vintage_scaler         = pickle.load( open ( self.home_path + 'src/features/vintage_scaler.pkl', 'rb' ))
        self.encode_region_code     = pickle.load( open ( self.home_path + 'src/features/encode_region_code.pkl', 'rb' ))
        self.encode_policy          = pickle.load( open ( self.home_path + 'src/features/encode_policy.pkl', 'rb' ))


    def rename_columns( df_train ):
        cols_old = df_train.columns

        cols_new = []
        cols_new = cols_old.map(lambda x: inflection.underscore(x))

        df_train.columns = cols_new
        
        return df_train

    
    def feature_engineering( self, df1 ):
        # vehicle_age
        df1['vehicle_age'] = df1['vehicle_age'].apply( lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_year' if x == '1-2 Year' else 'below_1_year')

        # vehicle_damage
        df1['vehicle_damage'] = df1['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0 )
        
        return df1
    
    def data_encoding( self, df3 ):
        # gender - Label Encoder
        gender_dict = {'Male': 0, 'Female': 1}
        df3['gender'] = df3['gender'].map( gender_dict )

        #region_code - Frequency Encoding 
        df3.loc[ : , 'region_code'] = df3['region_code'].map( self.encode_region_code )
        
        #vehicle_age - Ordinal Encoding  
        vehicle_age_dict = {'below_1_year': 1, 'between_1_2_year': 2, 'over_2_years': 3}
        df3['vehicle_age'] = df3['vehicle_age'].map( vehicle_age_dict )

        # policy_sales_channel  *Frequency Encoding 
        df3.loc[ : , 'policy_sales_channel'] = df3['policy_sales_channel'].map( self.encode_policy )

        return df3
    
    def data_rescaling( self, df3 ):
        #annual_premium
        df3['annual_premium'] = self.annual_premium_scaler.transform( df3[['annual_premium']].values )

        #age
        df3['age'] = self.age_scaler.transform( df3[['age']].values )
        
        #vintage
        df3['vintage'] = self.vintage_scaler.transform( df3[['vintage']].values )

        cols_selected = [ 'age', 'region_code', 'previously_insured', 'vehicle_damage', 'annual_premium','policy_sales_channel', 'vintage']

        return df3[ cols_selected ]
    
    def get_prediction( self, model, original_data, test_data):
        #model prediction
        pred = model.predict_proba( test_data)
        #join prediction into original data
        original_data['score'] = pred

        return original_data.to_json( orient='records' , date_format='iso')
        