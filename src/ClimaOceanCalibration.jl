module ClimaOceanCalibration

include("DiffusiveOceanCalibration.jl")
include("DataWrangling/DataWrangling.jl")
include("OMIPSimulations/OMIPSimulations.jl")
include("Visualization/Visualization.jl")

using .DiffusiveOceanCalibration
using .DataWrangling
using .OMIPSimulations
using .Visualization

end # module ClimaOceanCalibration

