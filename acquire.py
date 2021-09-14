import pandas as pd
import os

from env import host, username, password

def get_db_url(db_name, username=username, hostname=host, password=password):
    """ Builds URL for SQL query """
    return f'mysql+pymysql://{username}:{password}@{hostname}/{db_name}'

def zillow_query():
    """ Builds URL and query for zillow SQL query """
    url = get_db_url(db_name='zillow')
    query = """
            SELECT parcelid as ID,
                    transactiondate as DateSold,
                    taxvaluedollarcnt as Worth,
                    taxamount as Taxes,
                    roomcnt as Rooms,
                    bathroomcnt as Baths,
                    bedroomcnt as Beds,
                    garagecarcnt as GarageCarCount,
                    numberofstories as Stories,
                    lotsizesquarefeet as LotSize,
                    garagetotalsqft as GarageSize,
                    calculatedfinishedsquarefeet as FinishedSize,
                    yearbuilt as YearBuilt,
                    fips as LocalityCode,
                    regionidcounty as County,
                    regionidzip as Zipcode,
                    propertycountylandusecode as UseCode
            FROM properties_2017
            JOIN predictions_2017 USING(parcelid)
            WHERE propertylandusetypeid = 261 AND 
                  transactiondate BETWEEN '2017-05-01' AND '2017-08-31'
        """
    return query, url

def acquire_zillow():
    """ Returns zillow data, 
            queries database and stores data if none stored locally """
    if not os.path.isfile('zillow.csv'):
        query, url = zillow_query()
        df = pd.read_sql(query, url)
        df.to_csv('zillow.csv')
    return pd.read_csv('zillow.csv', index_col=0)
    
