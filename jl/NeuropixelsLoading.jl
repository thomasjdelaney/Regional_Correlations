push!(LOAD_PATH, homedir() * "/Julia_Utils/jl");

module NeuropixelsLoading

using JuliaUtils, MAT, NPZ, CSV, DataFrames, PyPlot, StatsBase, KernelDensity, Optim

export getClusterAveragePerSpike,
  getPeristimulusTimeHistogram,
  getRegionColourDict,
  getRegionDict,
  getRelevantTimesClusters,
  getTemplatePositionsAmplitudes,
  getXYcoords,
  makeBrainRegionPlot,
  makeCellInfoTable,
  plotCellResponseInfo,
  plotPeristimulusTimeHistogram,
  plotRasterForCell,
  plotRasterForProbe,
  plotStimuliResponse,
  plotStimulusOn,
  psthCost,
  readDataFromRegion,
  removeNoiseClusters

include("getClusterAveragePerSpike.jl")
include("getPeristimulusTimeHistogram.jl")
include("getRegionColourDict.jl")
include("getRegionDict.jl")
include("getRelevantTimesClusters.jl")
include("getTemplatePositionsAmplitudes.jl")
include("getXYcoords.jl")
include("makeBrainRegionPlot.jl")
include("makeCellInfoTable.jl")
include("plotCellResponseInfo.jl")
include("plotPeristimulusTimeHistogram.jl")
include("plotRasterForCell.jl")
include("plotRasterForProbe.jl")
include("plotStimuliResponse.jl")
include("plotStimulusOn.jl")
include("psthCost.jl")
include("readDataFromRegion.jl")
include("removeNoiseClusters.jl")

end
