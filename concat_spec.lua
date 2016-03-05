--[[ Code to concatenate the outputs of all specialists using a linear layer
This code is inspired by Sergey Zagoruyko, cf 
https://github.com/szagoruyko/cifar.torch ]]--

-- Imports
require 'xlua'
require 'optim'
require 'nn'
dofile 'provider.lua'
local c = require 'trepl.colorize'

-- Parameters
cmd = torch.CmdLine()
cmd:text('Train a master net')
cmd:text()
cmd:text('Options')
cmd:option('-input_dim', 120)
cmd:option('-save', 'logs_identity_concat_spec')
cmd:option('-domains', 'specialists/new.t7', 'Path to domains')
cmd:option('-batchSize', 10000)
cmd:option('-backend', 'cudnn')
cmd:option('-gpu', 'true')
cmd:option('-data', 'specialists')
cmd:text()

-- Parse input params
local opt = cmd:parse(arg)

-- Import cunn if GPU
if opt.gpu == 'true' then
  require 'cunn'
end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.fastest, cudnn.benchmark = true, true
end

-- Data loading
print(c.blue '==>' ..' loading data')
if string.find(opt.data, 'mnt') then
    os.execute('sudo chmod 777 ' .. opt.data)
end


provider = torch.load(opt.data .. '/specialist_scores_no_dust.t7')
provider.trainData.data = provider.trainData.data:float()
provider.valData.data = provider.valData.data:float()

confusion = optim.ConfusionMatrix(100)

print('Will save at '.. opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', 
                    '% mean class accuracy (test set)'}
testLogger.showPlot = false


print(c.blue'==>' ..' setting criterion')
if opt.gpu == 'true' then
  criterion = nn.CrossEntropyCriterion():cuda()
else
  criterion = nn.CrossEntropyCriterion()
end

if opt.gpu == 'true' then
  targets = torch.CudaTensor(opt.batchSize)
else
  targets = torch.FloatTensor(opt.batchSize)
end
local indices = torch.randperm(provider.trainData.data:size(1))
indices = indices:long():split(opt.batchSize)
-- Remove last element so that all the batches have equal size
--indices[#indices] = nil
-- /!\ ALWAYS HAVE A BATCHSIZE SUCH THAT BS | 40000

local tic = torch.tic()
-- Iterate over batches
for t,v in ipairs(indices) do
  xlua.progress(t, #indices)

  local outputs = provider.trainData.data:index(1,v)
  targets:copy(provider.trainData.label:index(1,v))

  confusion:batchAdd(outputs, targets)

end

confusion:updateValids()
print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
      confusion.totalValid * 100, torch.toc(tic)))
train_acc = confusion.totalValid * 100

confusion:zero()


print(c.blue '==>'.." testing")
local bs = opt.batchSize
for i=1,provider.valData.data:size(1),bs do
  local outputs = provider.valData.data:narrow(1,i,bs)
  confusion:batchAdd(outputs, provider.valData.label:narrow(1,i,bs))
end

confusion:updateValids()
print('Test accuracy:', confusion.totalValid * 100)

if testLogger then
  paths.mkdir(opt.save)
  testLogger:add{train_acc, confusion.totalValid * 100}
  testLogger:style{'-','-'}
  testLogger:plot()

  local base64im
  do
    cmd = 'convert -density 200 %s/test.log.eps %s/test.png'
    os.execute(cmd:format(opt.save,opt.save))
    cmd = 'openssl base64 -in %s/test.png -out %s/test.base64'
    os.execute(cmd:format(opt.save,opt.save))
    local f = io.open(opt.save..'/test.base64')
    if f then base64im = f:read'*all' end
  end

  -- Create HTML report
  -- Thanks to Sergey Zagoruyko, cf https://github.com/szagoruyko/cifar.torch
  local file = io.open(opt.save..'/report.html','w')
  file:write(([[
  <!DOCTYPE html>
  <html>
  <body>
  <title>%s - %s</title>
  <img src="data:image/png;base64,%s">
  <h4>optimState:</h4>
  <table>
  ]]):format(opt.save,epoch,base64im))
  for k,v in pairs(optimState) do
    if torch.type(v) == 'number' then
      file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
    end
  end
  file:write'</table><pre>\n'
  file:write(tostring(confusion)..'\n')
  file:write(tostring(model)..'\n')
  file:write'</pre></body></html>'
  file:close()
end

confusion:zero()

