module ClimaOceanCalibration

include("DiffusiveOceanCalibration.jl")
include("DataWrangling/DataWrangling.jl")
include("OMIPSimulations/OMIPSimulations.jl")

using .DiffusiveOceanCalibration
using .DataWrangling
using .OMIPSimulations

end # module ClimaOceanCalibration

