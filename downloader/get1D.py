from downloader import dataprocessing
from downloader import helper
import baspy as bp
import xarray as xr
import geopy
import os


class ClimateDataProcessing(object):

    def __init__(self, settings):

        # Make single value items into list
        exclusion_list = ['data_directory',
                          'filetype',
                          'variable_name_gcm',
                          'variable_name_era',
                          'variable_label',
                          ]
        for k, v in settings.items():
            if isinstance(v, str) and k not in exclusion_list:
                settings[k] = [v]

        self.settings = settings

        self.file_extension = {'netcdf': '.nc',
                               'df': '.csv',
                               }


    def save_clean_era_data(self, coords='default', verbose=False):
        """
        Process & save city timeseries data (ERA-Interim).

        Inputs:
        * coords in dictionary format if set manually; key='City', value={'longitude':X, 'latitude':Y}
        """

        if coords is 'default':
            _ = self.find_coords()
        else:
            self.coords = coords
            self.settings['cities'] = [city for city, _ in coords.items()]

        _ = self.get_era_data()

        self.get_cities(save=True, verbose=verbose)


    def save_clean_gcm_data(self, coords='default', verbose=False):
        """
        Process & save city timeseries data (GCM).

        Inputs:
        * coords in dictionary format if set manually; key='City', value={'longitude':X, 'latitude':Y}
        """

        if coords is 'default':
            _ = self.find_coords()
        else:
            self.coords = coords
            self.settings['cities'] = [city for city, _ in coords.items()]

        _ = self.get_gcm_catalogue()
        _ = self.get_gcm_data_from_catalogue()

        self.get_cities(save=True, verbose=verbose)


    def get_cities(self, save=True, verbose=False):
        """
        Get data arrays at city locations after global data is loaded
        """

        print("Saving data for all specified cities and RCP.")

        for i, city in enumerate(self.settings['cities']):

            if self.current_model is 'era':

                print("======= [{} ({}/{}), ERAI] =======".format(city,
                                                                  i + 1,
                                                                  len(self.settings['cities']),
                                                                  ))
                da_1D = self.get_data_at_location(city, verbose=verbose)
                if save:
                    self.save_data_at_location(da_1D)

            elif self.current_model is 'gcm':

                for j, rcp in enumerate(self.settings['future_rcp']):
                    print("======= [{} ({}/{}), {} ({}/{})] =======".format(city,
                                                            i + 1,
                                                            len(self.settings['cities']),
                                                            rcp,
                                                            j + 1,
                                                            len(self.settings['future_rcp'])))

                    da_1D = self.get_data_at_location(city, rcp, verbose=verbose)
                    if save:
                        self.save_data_at_location(da_1D)


    def get_gcm_catalogue(self):
        """
        Output: dataframe of baspy catalogue
        """

        print("GCM settings: model = {}, \
              rcp = {}, \
              start = {}, \
              end = {}".format(self.settings['model'],
                               self.settings['future_rcp'],
                               self.settings['model_start'],
                               self.settings['model_end'],
                               ))

        # Retrieve catalogue
        self.catalogue = bp.catalogue(dataset='cmip5',
                                      Model=self.settings['model'],
                                      Frequency='day',
                                      Experiment=['historical'] + self.settings['future_rcp'],
                                      RunID='r1i1p1',
                                      Var=self.settings['variable_name_gcm'],
                                      ).reset_index(drop=True)

        print(self.catalogue)

        return self.catalogue


    def get_gcm_data_from_catalogue(self, catalogue=None):
        """
        Output: dictionary of GCM data arrays (historical + RCP)
        """

        self.current_model = 'gcm'

        catalogue = self.catalogue if catalogue is None else catalogue

        # Get historical run
        print("Loading historical data array...(1/{})".format(len(self.settings['future_rcp']) + 1))
        idx_historical = catalogue['Experiment'].str.find('historical').idxmax()
        da_historical = bp.open_dataset(catalogue.iloc[idx_historical])[self.settings['variable_name_gcm']]
        da_historical = dataprocessing.slice_time(da_historical, self.settings['model_start'], self.settings['model_end'])

        # Get RCP runs
        self.gcm_data = {}
        for i, rcp in enumerate(self.settings['future_rcp']):

            print("Loading {} data array...({}/{})".format(rcp, i + 2, len(self.settings['future_rcp']) + 1))

            idx_rcp = catalogue['Experiment'].str.find(rcp).idxmax()
            da_rcp = bp.open_dataset(catalogue.iloc[idx_rcp])[self.settings['variable_name_gcm']]
            da_rcp = dataprocessing.slice_time(da_rcp, self.settings['model_start'], self.settings['model_end'])

            da_gcm = xr.concat([da_historical, da_rcp], dim='time')
            self.gcm_data[rcp] = da_gcm

        print("Data arrays loaded.")

        return self.gcm_data


    def get_era_data(self):
        """
        Output: ERA-Interim data array
        """

        self.current_model = 'era'

        era_data = dataprocessing.load_era(var=self.settings['variable_name_era'])

        self.era_data = dataprocessing.slice_time(era_data,
                                                  self.settings['observed_start'],
                                                  self.settings['observed_end'])

        print("ERA-Interim data loaded")

        return self.era_data


    def load_coords(self, filename='coords'):
        """
        Load coordinates from existing file
        Input:
        * filename = 'coords' (default)
        """

        try:
            self.coords = helper.open_pickle(self.settings['data_directory'] + filename)
            print("File found:")
            print(self.coords)
            return self.coords

        except FileNotFoundError as e:
            print("{}: \nUse find_coords() to get coordinate data".format(e))


    def find_coords(self, save_coords='default', filename='coords'):
        """
        Output: Dictionary of coordinates of cities given in settings
        """

        save_coords = self.settings['save_coords'] if save_coords is 'default' else save_coords

        cities = self.settings['cities']

        gc = geopy.geocoders.Nominatim(user_agent='user')
        self.coords = {}
        locations_not_found = []

        print("Geopy found the following locations:")
        for city in cities:
            city_geopy = gc.geocode(city)
            try:
                self.coords[city] = {'longitude': city_geopy.longitude, 'latitude': city_geopy.latitude}
                print("* '{}': {}".format(city, city_geopy))
            except AttributeError:
                locations_not_found.append(city)

        if locations_not_found:
            print("Not found: {}".format(locations_not_found))

        if save_coords:
            os.makedirs(self.settings['data_directory'], exist_ok=True)
            helper.save_pickle(self.coords, self.settings['data_directory'] + filename)
            print("You can load this later with self.load_coords(filename={})".format(filename))

        return self.coords


    def get_data_at_location(self, city, rcp=None, year_start=None, year_end=None, verbose=False):
        """
        Inputs: city, rcp (must be in self.settings)
        Output: processed & cleaned timeseries data array

        * Run first: self.get_gcm_data_from_catalogue(), self.find_coords()
        """

        if self.current_model is 'gcm':
            if rcp is None:
                raise AttributeError("rcp not specified")
            da = self.gcm_data[rcp]
        elif self.current_model is 'era':
            da = self.era_data

        coord = self.coords[city]
        da_1D = dataprocessing.preprocess_1d(da,
                                             lon=coord['longitude'],
                                             lat=coord['latitude'],
                                             year_start=year_start,
                                             year_end=year_end,
                                             GET_DF=False,
                                             verbose=verbose,
                                             NO_LEAP=self.settings['no_leap_years']
                                             )

        da_1D.attrs['city'] = city
        da_1D.attrs['rcp'] = str(rcp)
        da_1D.attrs['model_type'] = self.current_model

        return da_1D


    def save_data_at_location(self,
                              da_1D,
                              filetype='default',  # Default to settings
                              ):
        """
        Save data array
        """

        if filetype is 'default':
            filetype = self.settings['filetype']

        if self.current_model is 'gcm':
            file_label = da_1D.attrs['rcp']
        elif self.current_model is 'era':
            file_label = 'ERAI'

        save_location = (self.settings['data_directory']
                         + filetype + '/'
                         + self.settings['variable_label'] + '/')
        os.makedirs(save_location, exist_ok=True)  # does not re-create/overwrite directory

        name = '{}_{}{}'.format(da_1D.attrs['city'], file_label, self.file_extension[filetype])

        if filetype is 'netcdf':
            da_1D.to_netcdf(save_location + name)
        elif filetype is 'df':
            df = dataprocessing.da_to_df(da_1D)
            df.to_csv(save_location + name, index=False)

        print("Saved at {}".format(save_location + name))


    def load_data_at_location(self, city, model='era', rcp=None, filetype='default', save_location='default'):
        """
        Loads data if it exists in save_location
        Inputs:
        * city (e.g. 'London')
        * model ('era' or 'gcm')
        * rcp (speficy if loading GCM data e.g. 'rcp45')
        * filetype ('netcdf' or 'df')
        * save_location (specify search location; default is settings['data_directory'])
        """

        if filetype is 'default':
            filetype = self.settings['filetype']

        if save_location is 'default':
            save_location = (self.settings['data_directory']
                             + '{}/'.format(filetype)
                             + self.settings['variable_label'] + '/')

        # Giving inputs some leeway
        if rcp is not None:
            model = 'gcm'
        if rcp is 'era':
            model = 'era'
            rcp = None
        if isinstance(rcp, int):
            rcp = 'rcp{}'.format(rcp)
        if model is 'gcm':
            if rcp is None:
                raise AttributeError("Loading model='gcm', please specify rcp")
            filename_ext = rcp
        elif model is 'era':
            filename_ext = 'ERAI'
        else:
            raise AttributeError("model must be 'gcm' or 'era'")

        # Find file
        try:
            return xr.open_dataarray(save_location + '/{}_{}{}'.format(city, filename_ext, self.file_extension[filetype]))
        except FileNotFoundError as e:
            print("{}: \nCheck function arguments; or to get data, run save_clean_gcm_data()".format(e))
