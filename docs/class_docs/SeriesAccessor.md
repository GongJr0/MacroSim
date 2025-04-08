# SeriesAccessor
`SeriesAccessor` is a wrapper class built around the `fredapi` library to provide additional utilities that come in 
handy when working with time series of varying frequencies at the same time. Retrieving data through this class is 
not a necessity and `EqSearch` will essentially work with any arbitrary dataset that's suitable to regression. Also, 
note that you'll need a free API key to access FRED series, you can follow [this article](https://fred.stlouisfed.org/docs/api/api_key.html) to get one.

## Example Usage

```python
from macrosim import SeriesAccessor
import datetime as dt

fred = SeriesAccessor(
    key_path='./key.env',  # The class expects an .env file, the is loaded through environment variables
    key_name='fred_key'  # Name of the variable holding the API key in the .env file.
)

start = dt.datetime.fromisoformat("2000-01-01")
end = dt.datetime.fromisoformat("2024-01-01")

df = fred.get_series(series_ids=['CPIAUCSL', 'A264RX1A020NBEA', 'PSAVERT', 'M2REAL', 'GDPC1'], # FRED IDs of the series
                     date_range=(start, end), 
                     reindex_freq='QS', # Chosen time series frequency to reindex all data (defaults to max frequency available in the data
                     series_alias=['CPI', 'CAPINV','SRATE', 'M_2','RGDP']) # Col names to use in the output (this will change how variables are represented in equations)

df = fred.fill(
    df,
    [None, 'ffill', None, None, None] # Provide None, a built-in fill method, or a unary lambda function per column to fill the NaN values produced at reindexing
)
```

## Methods
`SeriesAccessor` includes two user-facing methods and it's only functionality is to retrieve and format series from FRED.

### `SeriesAccessor.get_series`
Retrieves the specified FRED series and applies the specified formatting.

__Params:__

- `series_ids: Sequence[str]`: List of FRED series IDs to retireve
- `date_range: tuple(dt.datetime.date, dt.datetime.date)`: Date interval to retrieve data for.
- `reindex_freq: str`: String literal of the observation frequency to reformat the series
- `series_alias: Sequence[str]`: List of aliases to use as column names in the outputted dataframe. `EqSearch`
will use these as the variable names when outputting symbolic equations.

__Returns:__

- `pd.DataFrame`: Concatenated and re-indexed dataframe of the series specified


### `SeriesAccessor.fill`
Fills the given data with specified fill methods.

__Params:__

- `df: pd.DataFrame`: DataFrame object to apply the fill methods
- `fill_methods`: a list of fill methods for each column in the dataframe. The default behavior is to fill the list with
`None`s until the amount of methods match the mount of columns in the data. The methods can be a unary lambda function, or
one of the built-in fill methods passed as a string. The available methods are:
  - `ffill`: forward-fill, behaves exactly same as `pd.Series.ffill`.
  - `bfill`: backward-fill, behaves exactly same as `pd.Series.bfill`.
  - `divide`: Divide the last known value equally to $n$ NaN values encountered before the next available row.
  For example, this can be used to evenly split yearly data to a monthly or quarterly observation frequency.
  - `mean`: Use the mean to fill NaNs.
  - `median`: Use the median to fill NaNs.
  - `IQR_mean`: Use the mean calculated from data within the IQR to fill NaNs.

__Returns:__

- `pd.DataFrame`: Dataframe with `NaN` values filled.