SELECT parcelid as ID,
	CASE
		WHEN fips = 6037 THEN 'Los Angeles County, CA' 
		WHEN fips = 6059 THEN 'Orange County, CA'
		WHEN fips = 6111 THEN 'Ventura County, CA'
	END AS Locality,
    transactiondate as DateSold,
	taxvaluedollarcnt as Worth,
    ROUND(taxamount / taxvaluedollarcnt, 3) as TaxRate,
    bathroomcnt as Baths,
   	bedroomcnt as Beds,
	lotsizesquarefeet as LotSize,
	calculatedfinishedsquarefeet as FinishedSize,
	(2017 - yearbuilt) as Age,
	propertylandusetypeid as "Type"
FROM properties_2017
JOIN predictions_2017 USING(parcelid)
WHERE propertylandusetypeid = 261 AND 
	transactiondate BETWEEN '2017-05-01' AND '2017-08-31'
ORDER BY DateSold ASC;