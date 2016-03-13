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
cmd:text('Train the final linear layer on concatenated scores')
cmd:text()
cmd:text('Options')
cmd:option('-input_dim', 209)
cmd:option('-save', 'logs_concat_spec')
cmd:option('-batchSize', 500)
cmd:option('-learningRate', 0.1)
cmd:option('-learningRateDecay', 1e-7)
cmd:option('-weightDecay', 0.005)
cmd:option('-momentum', 0.9)
cmd:option('-epoch_step', 25)
cmd:option('-max_epoch', 150)
cmd:option('-backend', 'cudnn')
cmd:option('-gpu', 'true')
cmd:option('-checkpoint', 25)
cmd:option('-data', 'specialists')
cmd:text()

-- Parse input params
local opt = cmd:parse(arg)

-- Import cunn if GPU
if opt.gpu == 'true' then
  require 'cunn'
end
-- Model configuration
print(c.blue '==>' ..' Configuring model')
local model = nn.Sequential()
if opt.gpu == 'true' then
  model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'):cuda())
  model:add(nn.Linear(opt.input_dim,100)):cuda()
else
  model:add(nn.Copy('torch.FloatTensor', 'torch.FloatTensor'))
  model:add(nn.Linear(opt.input_dim,100))
end
-- model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.fastest, cudnn.benchmark = true, true
   cudnn.convert(model:get(2), cudnn)
end

-- Data loading
print(c.blue '==>' ..' Loading data')
if string.find(opt.data, 'mnt') then
    os.execute('sudo chmod 777 ' .. opt.data)
end


provider = torch.load(opt.data .. '/specialist_scores.t7')
provider.trainData.data = provider.trainData.data:float()
provider.valData.data = provider.valData.data:float()

confusion = optim.ConfusionMatrix(100)

print('Will save at '.. opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', 
                    '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters, gradParameters = model:getParameters()

print(c.blue'==>' ..' Setting criterion')
if opt.gpu == 'true' then
  criterion = nn.CrossEntropyCriterion():cuda()
else
  criterion = nn.CrossEntropyCriterion()
end

print(c.blue'==>' ..' Configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

function train()
  -- Swith to train mode (flips, dropout, normalization)
  model:training()
  epoch = epoch or 1

  -- Drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then 
    optimState.learningRate = optimState.learningRate / 2 
  end
  
  print(c.blue '==>'.." Epoch # " .. 
    epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  if opt.gpu == 'true' then
    targets = torch.CudaTensor(opt.batchSize)
  else
    targets = torch.FloatTensor(opt.batchSize)
  end
  local indices = torch.randperm(provider.trainData.data:size(1))
  indices = indices:long():split(opt.batchSize)

  local tic = torch.tic()
  -- Iterate over batches
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.label:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)
      -- Add results to confusion matrix
      confusion:batchAdd(outputs, targets)

      return f, gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))
  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function test()
  -- Switch to test mode
  model:evaluate()
  print(c.blue '==>'.." Testing")
  local bs = 125
  for i=1,provider.valData.data:size(1),bs do
    local outputs = model:forward(provider.valData.data:narrow(1,i,bs))
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

  -- Save model every 'checkpoint' epochs
  if epoch % opt.checkpoint == 0 then
    local filename = paths.concat(opt.save, 'model' .. epoch .. '.net')
    print(c.blue '==>' .. 'Saving model to '.. filename)
    torch.save(filename, model:get(2):clearState())
  end

  confusion:zero()
end

--> Actual training script
for i=1,opt.max_epoch do
  train()
  test()
end
