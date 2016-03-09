--[[ Create and train specialist networks. ]]--

-- Imports
require 'xlua'
require 'optim'
require 'nn'
dofile 'provider.lua'
dofile 'unsupervised_provider.lua'
dofile 'custom_criterion.lua'
local c = require 'trepl.colorize'

-- Parameters
cmd = torch.CmdLine()
cmd:text('Train specialist networks')
cmd:text()
cmd:text('Options')
cmd:option('-model', 'vgg_specialists')
cmd:option('-save', 'specialist_logs')
cmd:option('-domains', 'specialists/new.t7')
cmd:option('-index', 1)
cmd:option('-data', 'default')
cmd:option('-batchSize', 128)
cmd:option('-learningRate', 75)
cmd:option('-learningRateDecay', 1e-6)
cmd:option('-weightDecay', 0.0000)
cmd:option('-momentum', 0.9)
cmd:option('-epoch_step', 20)
cmd:option('-max_epoch', 130)
cmd:option('-backend', 'cudnn')
cmd:option('-gpu', 'true')
cmd:option('-checkpoint', 130)
cmd:option('-alpha', 0.999, 'High temperature coefficient for knowledge transfer')
cmd:option('-T', 20, 'Temperature for knowledge transfer')
cmd:option('-unsupervised', false, 'Enable unsupervised learning')
cmd:option('-unsup_epochs', 50, 'Number of unsupervised learning epochs')
cmd:option('-unsup_data', 'default')
cmd:option('-verbose','false', 'print informaiton about the criterion')
cmd:option('-m','none', 'Add info to be included in the report.html')
cmd:text()

-- Parse input params
local opt = cmd:parse(arg)
opt.verbose = (opt.verbose == 'true')
if opt.data == 'default' then
  opt.data = '/mnt/specialist' .. opt.index .. '_provider.t7'
  opt.unsup_data = '/mnt/specialist' .. opt.index .. '_uprovider.t7'
end

-- Import cunn if GPU
if opt.gpu == 'true' then
  require 'cunn'
end

if opt.backend == 'cudnn' then
  require 'cudnn'
  cudnn.fastest, cudnn.benchmark = true, true
end

-- Data augmentation
-- Thanks to Sergey Zagoruyko, cf https://github.com/szagoruyko/cifar.torch
do 
  local BatchFlip, parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end

-- Specialist creation
print(c.blue '==>' ..' Creating specialist ' .. opt.index)
domains = torch.load(opt.domains)
domain = domains[opt.index]
num_class_specialist = #domain + 1
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
if opt.gpu == 'true' then
  model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'):cuda())
  model:add(dofile('specialists/' .. opt.model .. '.lua'):cuda())
else
  model:add(nn.Copy('torch.FloatTensor', 'torch.FloatTensor'))
  model:add(dofile('specialists/' .. opt.model .. '.lua'))
end
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   cudnn.convert(model:get(3), cudnn)
end

-- Data loading
print(c.blue '==>' ..' Loading data')
provider = torch.load(opt.data)
provider.trainData.data = provider.trainData.data:float()
provider.valData.data = provider.valData.data:float()

confusion = optim.ConfusionMatrix(num_class_specialist)

print('Will save at '.. opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', 
                    '% mean class accuracy (test set)'}
testLogger.showPlot = false

--[[ TODO !!!!!! ]]
parameters, gradParameters = model:getParameters()

print(c.blue'==>' ..' Setting criterion')
if opt.gpu == 'true' then
  criterion = DarkKnowledgeCriterion(opt.alpha, opt.T, opt.verbose):cuda()
else
  criterion = DarkKnowledgeCriterion(opt.alpha, opt.T, opt.verbose)
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
    epoch .. ' [batchSize = ' .. opt.batchSize .. '] specialist ' .. opt.index)

  targets = {}
  if opt.gpu == 'true' then
    targets.labels = torch.CudaTensor(opt.batchSize)
    targets.scores = torch.CudaTensor(opt.batchSize, num_class_specialist)
  else
    targets.labels = torch.FloatTensor(opt.batchSize)
    targets.scores = torch.FloatTensor(opt.batchSize, num_class_specialist)
  end
  local indices = torch.randperm(provider.trainData.data:size(1))
  indices = indices:long():split(opt.batchSize)
  -- Remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  -- Iterate over batches
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
    targets.labels:copy(provider.trainData.label:index(1,v))
    targets.scores:copy(provider.trainData.scores:index(1,v))
    
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)
      -- Add results to confusion matrix
      confusion:batchAdd(outputs, targets.labels)

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
    if i> provider.valData.data:size(1)-bs then
	i = provider.valData.data:size(1)-bs
    end
    local outputs = model:forward(provider.valData.data:narrow(1,i,bs))
    local labels = provider.valData.label:narrow(1,i,bs)
    confusion:batchAdd(outputs, labels)
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
    local file = io.open(opt.save..'/report' .. opt.index .. '.html','w')
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
    file:write('<tr><td>Temp</td><td>'.. opt.T ..'</td></tr>\n')
    file:write('<tr><td>alpha</td><td>'.. opt.alpha ..'</td></tr>\n')
    file:write('<tr><td>initial LR</td><td>'.. opt.learningRate ..'</td></tr>\n')
    file:write('<tr><td>Comments</td><td>'.. opt.m ..'</td></tr>\n')
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- Save model every 'checkpoint' epochs
  if epoch % opt.checkpoint == 0 then
    local model_name = 'sp' .. opt.index .. 'ep' .. epoch .. '.net'
    local filename = paths.concat(opt.save, model_name)
    print(c.blue '==>' .. 'Saving model to '.. filename)
    torch.save(filename, model:get(3):clearState())
  end

  confusion:zero()
end

-- Actual training script
for i=1,opt.max_epoch do
  train()
  test()
end

-- Unsupervised learning
function train_unsupervised()
  -- Swith to train mode (flips, dropout, normalization)
  model:training()
  epoch = epoch or 1

  -- Drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then 
    optimState.learningRate = optimState.learningRate / 2 
  end
  
  print(c.blue '==>'.." Unsupervised epoch # " .. 
    epoch .. ' [batchSize = ' .. opt.batchSize .. '] specialist ' .. opt.index)

  targets = {}
  if opt.gpu == 'true' then
    targets.labels = torch.CudaTensor(opt.batchSize)
    targets.scores = torch.CudaTensor(opt.batchSize, num_class_specialist)
  else
    targets.labels = torch.FloatTensor(opt.batchSize)
    targets.scores = torch.FloatTensor(opt.batchSize, num_class_specialist)
  end
  local indices = torch.randperm(uprovider.trainData.data:size(1))
  indices = indices:long():split(opt.batchSize)
  -- Remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  -- Iterate over batches
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = uprovider.trainData.data:index(1,v)
    targets.labels = uprovider.trainData.label
    targets.scores:copy(uprovider.trainData.scores:index(1,v))
    
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)
      -- Add results to confusion matrix
      confusion:batchAdd(outputs, targets.labels)

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

if opt.unsupervised == true then
  -- Data loading
  print(c.blue '==>' ..' Loading unsupervised data')
  uprovider = torch.load(opt.unsup_data)
  uprovider.trainData.data = uprovider.trainData.data:float()

  print(c.blue'==>' ..' Setting unsupervised criterion')
  if opt.gpu == 'true' then
    criterion = DarkKnowledgeCriterion(1.0, opt.T):cuda()
  else
    criterion = DarkKnowledgeCriterion(1.0, opt.T)
  end

  for i=1,opt.unsup_epochs do
    train_unsupervised()
    test()
  end
end
