---
--- Created by killa.
--- DateTime: 17-8-3 下午7:30
---

print("Start time: " .. os.time())

require 'nn'
require 'torch'
require 'optim'
require 'model.plaincnn'

local options = require 'options'

-- vocab structure:
--  {
--      idx2word: ["word1", "word2"],
--      idx2vec: Tensor,
--      word2idx: {
--          word1: idx,
--          word2: idx
--      },
--      word2count: {
--          word1: count,
--          word2: count
--      }
--  }

local epoch = options.epoch
local batchSize = options.batchSize
local lr = options.lr
local lrd = options.lrd
local gpu = options.gpu
local logInterval = options.logInterval
local maxAbsLength = options.maxAbsLength
local logFilePath = options.logFile

-- Load data
local rawDataset = torch.load('data/nostem.nopunc.case/discrete/ke20k_training.json.t7')
local validData = torch.load('data/nostem.nopunc.case/discrete/ke20k_validation.json.t7')
local vocab = torch.load('data/nostem.nopunc.case/ke20k.nostem.nopunc.case.vocab.t7')
local emb = vocab.idx2vec
local data = rawDataset.data
local label = rawDataset.label
local dataSize = #data
local validDataSize = batchSize -- TODO 强行固定eval data size

if gpu then
    require 'cunn'
    require 'cutorch'
    emb = emb:cuda()
    logFilePath = logFilePath.gsub('.log', '_cuda.log')
end


-- Build logger
local logger = optim.Logger('training.cuda.log')
logger:setNames{'training loss', 'validation loss'}
logger:style{'-', '-'}


-- Build training data
local batchedDataset = {}
local validDataset = {}

-- Batch 化数据
-- Training data
print("Making batch training data...")
local i = 1 -- Data index

while i <= dataSize do -- 每次生成一个batch
    local batchData = torch.LongTensor(batchSize * maxAbsLength):fill(1)
    local batchLabel = torch.DoubleTensor(batchSize, maxAbsLength):fill(0)

    for dataIndex = 1, batchSize do
        if i <= dataSize then
            local oneData = data[i]
            local oneLabel = label[i]
            local len = oneData:size(1)
            local padSize = maxAbsLength - len
            local startPos = 1 + math.floor(padSize / 2)
            local endPos = maxAbsLength - math.ceil(padSize / 2)
            local dataBase = (dataIndex - 1) * maxAbsLength

            batchData[{{dataBase + startPos, dataBase + endPos}}]:copy(oneData)
            batchLabel[dataIndex][{{startPos, endPos}}]:copy(oneLabel)
            i = i + 1
        else
            i = i + 1
            break
        end

        io.write('\riter: ' .. i)
        io.flush()
    end

    if gpu then
        table.insert(batchedDataset, {data = {emb, batchData:cuda()}, label = batchLabel:cuda()})
    else
        table.insert(batchedDataset, {data = {emb, batchData}, label = batchLabel})
    end
end
print("Finished batch data building.")

-- Validation data
print("Making validation data...")
local validBatchData = torch.LongTensor(validDataSize * maxAbsLength):fill(1)
local validBatchLabel = torch.DoubleTensor(validDataSize, maxAbsLength):fill(0)

i = 1
while i <= validDataSize do
    local oneData = validData.data[i]
    local oneLabel = validData.label[i]
    local len = oneData:size(1)
    local padSize = maxAbsLength - len
    local startPos = 1 + math.floor(padSize / 2)
    local endPos = maxAbsLength - math.ceil(padSize / 2)
    local dataBase = (i - 1) * maxAbsLength

    validBatchData[{{dataBase + startPos, dataBase + endPos}}]:copy(oneData)
    validBatchLabel[i][{{startPos, endPos}}]:copy(oneLabel)
    i = i + 1

    io.write('\riter: ' .. i)
    io.flush()
end

local validMask = torch.ne(validBatchData, 1):double():reshape(validDataSize, maxAbsLength)

if gpu then
    validDataset = {data = {emb, validBatchData:cuda()}, label = validBatchLabel:cuda(), mask = validMask:cuda(), num = validMask:sum()}
else
    validDataset = {data = {emb, validBatchData}, label = validBatchLabel, mask = validMask, num = validMask:sum()}
end
print('\rFinished validation data building.')
-- Batch 化数据

------------------------------------------------------------------------------------------------------------------------
-- Build MODEL
local model = plaincnn(options)

-- Build criterion
--local criterion = nn.MSECriterion()
local criterion = nn.AbsCriterion()

if gpu then
    model = model:cuda()
    criterion = criterion:cuda()
end
------------------------------------------------------------------------------------------------------------------------
print(model)

-- Trainging
params, gradParams = model:getParameters()
local optimConfig = {
    learningRate = lr,
    learningRateDecay = lrd,
    weightDecay = 10,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-08
}
local lossFactor = batchSize * maxAbsLength

for iter = 1, epoch do
    print('\nTraining epoch ' .. iter .. '\n')

    for i, v in ipairs(batchedDataset) do
        io.write('\rBatch number: ' .. i)
        io.flush()
        local mask = torch.ne(v.data[2], 1):double():reshape(batchSize, maxAbsLength)
        local num = mask:sum() -- Real Abstract length

        if gpu then
            mask = mask:cuda()
        end

        function feval(params)
            gradParams:zero()

            local outputs = model:forward(v.data)
            outputs:cmul(mask)

            assert(outputs:size(1) == v.label:size(1))

            local loss = criterion:forward(outputs, v.label)
            local dloss_doutputs = criterion:backward(outputs, v.label)
            model:backward(v.data, dloss_doutputs)

            return loss, gradParams
        end

        _, l = optim.adam(feval, params, optimConfig)

        io.write(' Learning Rate: ' .. lr / (1 + i * lrd))
        io.flush()

        if i % math.floor(logInterval / batchSize) == 0 then
            -- Log loss and plot
            local validationOutput = model:forward(validDataset.data)
            validationOutput:cmul(validDataset.mask)

            local validationLoss = criterion:forward(validationOutput, validDataset.label)
            validationLoss = validationLoss / validDataset.num

            local lossPair = {lossFactor * l[1] / num, lossFactor * validationLoss}

            logger:add(lossPair)
            --logger:plot()
        end
    end
end

torch.save('../trainings/model.t7', {options = options, model = model})

print("End time: " .. os.time())