---
--- Created by killa.
--- DateTime: 17-8-3 下午7:30
---

require 'torch'

cmd = torch.CmdLine()

cmd:text()
cmd:text('Training a model')
cmd:text()
cmd:text('Options')
cmd:option('-epoch', 10, 'How many epochs to train. [10]')
cmd:option('-batchSize', 128, 'Size of each batch. [128]')
cmd:option('-lr', 0.001, 'Learning rate. [0.001]')
cmd:option('-lrd', 0.05, 'Learning rate decay. lr = lr/(1+step*lrd). [0.05]')
cmd:option('-gpu', false, 'Whether use gpu. [false]')
cmd:option('-kernalWidth', 7, 'Width of convolutional kernal. [7]')
cmd:option('-convLayer', 4, 'How many layers of convolution. [4]')
cmd:option('-logInterval', 1000, 'Loss will be logged every [1000] steps.')
cmd:option('-fineTune', false, 'Whether fine tune the embedding of words. [false]')
cmd:option('-maxAbsLength', 500, 'Maximum length of passage. [500]')
cmd:option('-embDimension', 300, 'Word embedding dimension. [300]')
cmd:option('-channelSize', 500, 'Channel size of convolution. [500]')
cmd:option('-logFile', 'trainings/' .. os.date('%m%d/%H-%M-%S.log'), 'Logging file. [trainings/{datetime}_{cuda}.log]')
cmd:option('-plot', false, 'Whether plot loss curve during training. Can cause problem. [false]')
cmd:option('-save', 'trainings/model.t7', 'Save path of trained model. [trainings/model.t7]')
cmd:text()

-- parse input params
params = cmd:parse(arg)

return params