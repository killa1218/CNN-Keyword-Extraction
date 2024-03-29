---
--- Created by killa.
--- DateTime: 17-8-12 上午10:58
---

require 'torch'
require 'nn'
require 'utils'
require 'cutorch'
require 'cunn'

local cmd = torch:CmdLine()

cmd:option('-model', 'trainings/model.t7', 'Model path.')
cmd:option('-data', 'data/nostem.nopunc.case/discrete/ke20k_testing.json.t7', 'Path of test data.')
cmd:option('-top', 10, 'Get top [top] words to be result.')
cmd:option('-sample', 'test/test_sample.txt', 'Sample test result path.')
cmd:option('-vocab', 'data/nostem.nopunc.case/ke20k.nostem.nopunc.case.vocab.t7', 'Vocabulary file path.')

local params = cmd:parse(arg)

local modelInfo = torch.load(params.model)
local model = modelInfo.model
local options = modelInfo.options

local batchSize = options.batchSize
local maxAbsLength = options.maxAbsLength
local rawTestData = torch.load(params.data)
local vocab = torch.load(params.vocab)
local emb = vocab.idx2vec
emb = emb:cuda()
local i2w = vocab.idx2word
local testResult = params.sample
local f = io.open(testResult, 'w')

local batchedTestData = makeBatchData(rawTestData, emb, options)
local testDataNum = #batchedTestData

local totalRightNum = 0
local totalLabelNum = 0
local totalDataNum = 0
local coveredDataNum = 0

for i, v in ipairs(batchedTestData) do
    io.write(string.format('\rBatch number: %d/%d', i, testDataNum))
    io.flush()

    local dataMask = torch.gt(v.data[2], 1):double():reshape(batchSize, maxAbsLength)
    local groundTruthMask = torch.eq(v.label, 1):double()
    local randomPick = math.ceil(math.random() * batchSize)

    if options.gpu then
        dataMask = dataMask:cuda()
        groundTruthMask = groundTruthMask:cuda()
        model:cuda()
    end

    local output = model:forward(v.data):cmul(dataMask)
    local topIdx
    _, topIdx = output:topk(params.top, true) -- Find the index of top 10 scores

    assert(topIdx:size()[1] == options.batchSize, '[ERROR] topIdx line number wrong.')
    assert(topIdx:size()[2] == params.top, '[ERROR] top number wrong.')

    -- Log the random selected sample test data to a file. (each batch one sample)
    local sampleData = v.data[2]:reshape(batchSize, maxAbsLength)[randomPick]
    local sampleLabel = torch.cmul(sampleData, groundTruthMask[randomPick]) -- Contains indexs of keywords
    local sampleOutput = torch.cmul(output[randomPick], dataMask[randomPick])
    local sampleTopIdx = topIdx[randomPick]
    local keywords = {}

    f:write('Passage: ')
    for j = 1, sampleData:size()[1] do
        local idx = sampleData[j]
        local groundTruth = sampleLabel[j]

        if idx > 1 then
            assert(sampleOutput[j] ~= 0, '[ERROR] position wrong.')
            f:write(string.format('%s(%.3f) ', i2w[idx], sampleOutput[j]))
        end

        if groundTruth > 1 then
            table.insert(keywords, i2w[groundTruth])
        end
    end
    f:write('\n') -- End writing labeled passage

    f:write('Key words: ')
    for _, k in pairs(keywords) do
        f:write(string.format('%s ', k))
    end
    f:write('\n') -- End writing ground truth

    f:write(string.format('Top %d words: ', params.top))
    for ii = 1, sampleTopIdx:size()[1] do
        local k = sampleTopIdx[ii]
        f:write(string.format('%s ', i2w[sampleData[k]]))
    end
    f:write('\n\n') -- End writing one sample


    -- Count right predictions
    local matchMap = groundTruthMask:gather(2, topIdx) -- Which predictions are in ground truth
    local rightNum = matchMap:sum()

    totalRightNum = totalRightNum + rightNum
    totalLabelNum = totalLabelNum + groundTruthMask:sum()

    local dataInThisBatch = torch.ne(groundTruthMask:sum(2), 0):int():sum()
    local coveredDataInThisBatch = torch.ne(matchMap:sum(2), 0):int():sum()

    totalDataNum = totalDataNum + dataInThisBatch
    coveredDataNum = coveredDataNum + coveredDataInThisBatch
end

local precision = totalRightNum / totalLabelNum
local recall = coveredDataNum / totalDataNum
local f1 = 2 * precision * recall / (precision + recall)

print(string.format('\nPrecision: %.4f', precision))
print(string.format('Recall: %.4f', recall))
print(string.format('F1 score: %.4f', f1))

f:write(string.format('Precision: %.4f\nRecall: %.4f\nF1 score: %.4f\n', precision, recall, f1))
f:close()
