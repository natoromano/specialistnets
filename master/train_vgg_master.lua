--[[ Code to train a master VGGNet on CIFAR-100 (or any training data in a
provider file).

This code is widely inspired by Sergey Zagoruyko, 
cf https://github.com/szagoruyko/cifar.torch ]]--

require 'xlua'
require 'optim'
local c = require 'trepl.colorize'

-- Parameters
opt = {save='logs', batchSize=128, learningRate=1, learningRateDecay=1e-7, 
       weightDecay=0.0005, momentum=0.9, epoch_step=25, model='vgg_cifar_100', 
       max_epoch=150, backend='cudnn', gpu=true, checkpoint=25}
print ('PARAMETERS')
print(opt)

if opt.gpu == 'true' then
	require 'cunn'
end


-- Data augmentation
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


-- Model configuration
print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
if gpu == true then
	model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'):cuda())
	model:add(dofile('master/' .. opt.model .. '.lua'):cuda())
else
	model:add(nn.Copy('torch.FloatTensor', 'torch.FloatTensor'))
	model:add(dofile('master/' .. opt.model .. '.lua'))
end
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(3), cudnn)
end


-- Data loading
print(c.blue '==>' ..' loading data')
provider = torch.load 'master/master_provider.t7'
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

print(c.blue'==>' ..' setting criterion')
if opt.gpu == true then
	criterion = nn.CrossEntropyCriterion():cuda()
else
	criterion = nn.CrossEntropyCriterion()
end

print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- Drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then 
  	optimState.learningRate = optimState.learningRate/2 end
  -- Save model every "checkpoint" epochs
  if epoch % opt.checkpoint == 0 then 
    torch.save('model' .. epoch .. '.t7', model) 
  end
  
  print(c.blue '==>'.." online epoch # " .. 
  	epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  if opt.gpu == 'true' then
  	targets = torch.CudaTensor(opt.batchSize)
  else
  	targets = torch.FloatTensor(opt.batchSize)
  end
  local indices = torch.randperm(provider.trainData.data:size(1))
  indices = indices:long():split(opt.batchSize)
  -- Remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
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
  -- Disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1, provider.valData.data:size(1),bs do
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

  -- save model every 50 epochs
  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3):clearState())
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  test()
end