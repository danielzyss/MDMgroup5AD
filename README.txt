CODE FOR MDM-2 GROUP 5: ALZHEIMER CONNECTOME

please have the data as a pandas pickled dataframe in a '/data' folder. The arrangement of the dataframe must be such that
every column is an electrode, and every row a patient such that the row 0, column 0 contains the numpy time series of
patient 0 at electrode 'Fp1' for example (The dataframe can be generated from the original matlab files using the
matlabTranslator.py file, with the function MatlabStructure2PandasDataframe() ) .

Each model can be run separately from the main.py file, or any function can be called independently from the tool.py file. Every
parameter is modifiable to try out different results.

Please read the report for more explaination.

Best,

Group 5