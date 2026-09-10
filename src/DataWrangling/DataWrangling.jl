module DataWrangling

export TimeAverageOperator, TimeAverageBuoyancyOperator, AveragedFieldTimeSeries, regrid_levels!, spatial_averaging, save_averaged_fieldtimeseries

include("fieldtimeseries_averaging.jl")

end