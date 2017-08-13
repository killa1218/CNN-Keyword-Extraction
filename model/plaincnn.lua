---
--- Created by killa.
--- DateTime: 17-8-12 上午10:08
---

require 'nn'
require 'torch'

function plaincnn(options)
    local batchSize = options.batchSize
    local maxAbsLength = options.maxAbsLength
    local embDimension = options.embDimension
    local kernalWidth = options.kernalWidth
    local channelSize = options.channelSize
    local convLayer = options.convLayer

    -- Build MODEL
    -- MODEL Build index
    local modelLookup = nn.Sequential()
    modelLookup:add(nn.Index(1))

    -- MODEL Build reshape
    local modelReshape = nn.Reshape(batchSize, maxAbsLength, embDimension)

    -- MODEL Build padding
    local leftPadSize = math.floor((kernalWidth - 1) / 2)
    local rightPadSize = kernalWidth - 1 - leftPadSize
    local modelPadding = nn.Sequential():add(nn.Padding(2, -leftPadSize)):add(nn.Padding(2, rightPadSize))

    -- MODEL Build convolution
    local modelConvolution = nn.Sequential()
    --modelConvolution:add(nn.TemporalConvolution(300, 1, kernalWidth)):add(nn.Sigmoid())
    --modelConvolution:add(nn.TemporalConvolution(300, channelSize, kernalWidth)):add(nn.Sigmoid())
    modelConvolution:add(nn.TemporalConvolution(300, channelSize, kernalWidth)):add(nn.LeakyReLU())
    for i = 1, convLayer - 3 do
        local localPad = nn.Sequential():add(nn.Padding(2, -leftPadSize)):add(nn.Padding(2, rightPadSize))
        modelConvolution:add(localPad):add(nn.TemporalConvolution(channelSize, channelSize, kernalWidth)):add(nn.LeakyReLU())
        --modelConvolution:add(localPad):add(nn.TemporalConvolution(channelSize, channelSize, kernalWidth)):add(nn.Sigmoid())
    end

    local finalPad = nn.Sequential():add(nn.Padding(2, -leftPadSize)):add(nn.Padding(2, rightPadSize))
    modelConvolution:add(finalPad):add(nn.TemporalConvolution(channelSize, math.floor(channelSize / 2), kernalWidth)):add(nn.LeakyReLU())
    --modelConvolution:add(finalPad):add(nn.TemporalConvolution(channelSize, math.floor(channelSize / 2), kernalWidth)):add(nn.Sigmoid())
    finalPad = nn.Sequential():add(nn.Padding(2, -leftPadSize)):add(nn.Padding(2, rightPadSize))
    modelConvolution:add(finalPad):add(nn.TemporalConvolution(math.floor(channelSize / 2), 1, kernalWidth)):add(nn.Sigmoid())

    -- MODEL Build whole model
    local model = nn.Sequential()
    local padding = nn.Sequential():add(modelLookup):add(modelReshape):add(modelPadding)
    model:add(padding):add(modelConvolution):add(nn.Reshape(batchSize, maxAbsLength))

    return model
end

return plaincnn