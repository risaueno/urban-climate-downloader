#%%
settings = {
    # --------------------------- #
    #  Location / variable        #
    # --------------------------- #
    'cities': ['London', 'NYC', 'Beijing', 'Tokyo', 'Madrid'],  # Cities to download

    'variable_name_gcm': 'tasmax',  # GCM official variable name e.g. 'tas' (mean T), 'tasmax' (max T), 'tasmin' (min T)
    'variable_name_era': 't2max',   # ERAI official variable name e.g. 'T2' (mean T), 't2max' (max T), 't2min' (min T)
    'variable_label': 'max_temperature',  # Downloaded data will go in this folder; ensure accurate discription

    'no_leap_years': True,   # Remove 29th Feb's to get 365-days consistent calender

    # --------------------------- #
    #  GCM                        #
    # --------------------------- #
    'model': 'HadGEM2-CC',  # Climate model
    'future_rcp': ['rcp85', 'rcp45'],  # Model RCP
    'model_start': 1980,  # Model start year (inclusive)
    'model_end': 2050,  # Model end year (inclusive)

    # --------------------------- #
    #  ERA Interim                #
    # --------------------------- #
    'observed_start': 1980,  # ERA Interim start year (inclusive)
    'observed_end': 2017,  # ERA Interim end year (inclusive)

    # --------------------------- #
    #  Saving files               #
    # --------------------------- #
    'save_coords': True,  # Optional: save loaded coordinates from cities
    'filetype': 'netcdf',  # Climate data type to save 'netcdf' or 'df'
    'data_directory': './data/riskindex/'  # Directory to save created data
}

#%%
import sys
print(sys.path)

from downloader.get1D import ClimateDataProcessing
cd = ClimateDataProcessing(settings)

#%%
cd.save_clean_gcm_data()
cd.save_clean_era_data()
