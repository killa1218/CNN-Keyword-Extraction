---
--- Created by killa.
--- DateTime: 17-8-12 上午10:22
---

require 'torch'

function table.slice(tbl, first, last, step)
    ---
    ---Get a slice of the whole table
    ---
    local sliced = {}

    for i = first or 1, last or #tbl, step or 1 do
        sliced[#sliced+1] = tbl[i]
    end

    return sliced
end

function makeBatchData(rawData, emb, options)
    local batchedDataset = {}
    local i = 1 -- Data index
    local maxAbsLength = options.maxAbsLength
    local batchSize = options.batchSize
    local gpu = options.gpu
    local data = rawData.data
    local label = rawData.label
    local dataSize = #data

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

    return batchedDataset
end
