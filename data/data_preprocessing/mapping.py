import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer

def map_categorical_columns(df):
    df = df.copy()
    city_map = {
        'Mumbai': 'Metro India', 'Thane & Outskirts': 'Metro India', 'Other Metro Cities': 'Metro India',
        'Other Cities of Maharashtra': 'Tier II India', 'Tier II Cities': 'Tier II India', 'Other Cities': 'Other India',
        'Select': 'Unknown', 'nan': 'Unknown', pd.NA: 'Unknown', None: 'Unknown'
    }
    country_map = {
        'India': 'India', 'United States': 'North America', 'Canada': 'North America',
        'United Arab Emirates': 'Middle East', 'Saudi Arabia': 'Middle East', 'Qatar': 'Middle East', 'Kuwait': 'Middle East',
        'Oman': 'Middle East', 'Bahrain': 'Middle East',
        'Germany': 'Europe', 'France': 'Europe', 'United Kingdom': 'Europe', 'Sweden': 'Europe',
        'Belgium': 'Europe', 'Netherlands': 'Europe', 'Switzerland': 'Europe',
        'China': 'Asia', 'Singapore': 'Asia', 'Hong Kong': 'Asia', 'Philippines': 'Asia', 'Vietnam': 'Asia',
        'South Africa': 'Africa', 'Nigeria': 'Africa',
        'nan': 'Unknown', 'unknown': 'Other'
    }
    specialization_map = {
        'Marketing Management': 'Marketing', 'Operations Management': 'Operations', 'Finance Management': 'Finance',
        'Human Resource Management': 'HR', 'International Business': 'Business', 'Business Administration': 'Business', 'MBA': 'Business',
        'IT Projects Management': 'IT', 'E-Business': 'E-Commerce', 'E-Commerce': 'E-Commerce', 'E Commerce': 'E-Commerce',
        'Supply Chain Management': 'Operations', 'Retail Management': 'Operations', 'Banking, Investment And Insurance': 'Finance',
        'Healthcare Management': 'Healthcare', 'Hospitality Management': 'Healthcare',
        'Rural and Agribusiness': 'Other', 'Travel and Tourism': 'Other', 'Media and Advertising': 'Other', 'Services Excellence': 'Other',
        'Not Specified': 'Unknown', 'Select': 'Unknown', 'nan': 'Unknown', None: 'Unknown'
    }
    education_source_map = {
        'Online Search': 'Digital', 'Advertisements': 'Digital', 'Email': 'Digital', 'SMS': 'Digital', 'Social Media': 'Digital',
        'Word Of Mouth': 'Referral', 'Student of SomeSchool': 'Referral',
        'Multiple Sources': 'Multi-Channel', 'Other': 'Other', 'Select': 'Unknown', 'nan': 'Unknown', None: 'Unknown'
    }
    occupation_map = {
        'Student': 'Student', 'Working Professional': 'Working', 'Businessman': 'Working',
        'Housewife': 'Non-Working', 'Unemployed': 'Non-Working', 'Other': 'Unknown', 'Select': 'Unknown', 'nan': 'Unknown', None: 'Unknown'
    }
    tags_map = {
        'Will revert after reading the email': 'Pending Response', 'Still Thinking': 'Pending Response', 'Interested in full time MBA': 'Pending Response',
        'Ringing': 'Trying to Contact', 'Busy': 'Trying to Contact', 'switched off': 'Trying to Contact', 'opp hangup': 'Trying to Contact',
        'Interested in other courses': 'Not Interested', 'Already a student': 'Not Interested', 'Lost to EINS': 'Not Interested', 'Lost to Others': 'Not Interested', 'Not doing further education': 'Not Interested',
        'invalid number': 'Invalid Contact', 'wrong number given': 'Invalid Contact', 'number not provided': 'Invalid Contact',
        'Diploma holder (Not Eligible)': 'Not Eligible', 'Graduation in progress': 'Not Eligible',
        'Closed by Horizzon': 'Converted', 'Want to take admission but has financial problems': 'Financial Issue', 'in touch with EINS': 'Transferred'
    }
    lead_quality_map = {
        'High in Relevance': 'High', 'Might be': 'Medium', 'Not Sure': 'Medium', 'Low in Relevance': 'Low', 'Worst': 'Low', 'nan': 'Unknown', None: 'Unknown'
    }
    lead_profile_map = {
        'Potential Lead': 'Prospective', 'Other Leads': 'Prospective', 'Student of SomeSchool': 'Converted',
        'Lateral Student': 'Converted', 'Dual Specialization Student': 'Converted', 'Select': 'Unknown', 'nan': 'Unknown', None: 'Unknown'
    }

    mappings = {
        'City': city_map,
        'Country': country_map,
        'Specialization': specialization_map,
        'How did you hear about X Education': education_source_map,
        'What is your current occupation': occupation_map,
        'Tags': tags_map,
        'Lead Quality': lead_quality_map,
        'Lead Profile': lead_profile_map
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: mapping.get(x, 'Unknown') if pd.notna(x) else 'Unknown')
            df[col] = df[col].fillna('Unknown')
    return df

mapping_transformer = FunctionTransformer(map_categorical_columns, validate=False)
