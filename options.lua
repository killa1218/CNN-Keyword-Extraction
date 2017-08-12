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
cmd:option('-epoch', 10, 'How many epochs to train.')
cmd:option('-batchSize', 128, 'Size of each batch.')
cmd:option('-lr', 0.001, 'Learning rate.')
cmd:option('-lrd', 0.05, 'Learning rate decay. lr = lr/(1+step*lrd).')
cmd:option('-gpu', false, 'Whether use gpu.')
cmd:option('-kernalWidth', 7, 'Width of convolutional kernal.')
cmd:option('-convLayer', 4, 'How many layers of convolution.')
cmd:option('-logInterval', 1000, 'Loss will be logged every [1000] steps.')
cmd:option('-fineTune', false, 'Whether fine tune the embedding of words.')
cmd:option('-maxAbsLength', 500, 'Maximum length of passage.')
cmd:option('-embDimension', 300, 'Word embedding dimension.')
cmd:option('-channelSize', 500, 'Channel size of convolution.')
cmd:option('-logFile', 'trainings/' .. os.date('%m%d/%H-%M-%S.log'), 'Logging file.')
cmd:option('-plot', false, 'Whether plot loss curve during training. Can cause problem.')
cmd:option('-save', 'trainings/model.t7', 'Save path of trained model.')
cmd:text()

-- parse input params
params = cmd:parse(arg)

return params