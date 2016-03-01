--[[ Create and train specialist networks. ]]--

-- Imports
require 'xlua'
require 'optim'
require 'nn'
dofile 'provider.lua'
dofile 'custom_criterion.lua'
local c = require 'trepl.colorize'

-- Parameters
cmd = torch.CmdLine()
cmd:text('Train spcialist networks')
cmd:text()
cmd:text('Options')
cmd:option('-model', 'vgg_specialists')
cmd:option('-save', 'specialist_logs')
cmd:option('-domains', 'specialists/domains.t7')
cmd:option('-index', 1)
cmd:option('-data', '/mnt/specialist_provider.t7')
cmd:option('-batchSize', 128)
cmd:option('-learningRate', 1)
cmd:option('-learningRateDecay', 1e-7)
cmd:option('-weightDecay', 0.0005)
cmd:option('-momentum', 0.9)
cmd:option('-epoch_step', 25)
cmd:option('-max_epoch', 150)
cmd:option('-backend', 'cudnn')
cmd:option('-gpu', 'true')
cmd:option('-checkpoint', 25)
cmd:option('-alpha', 0.9, 'High temperature coefficient for knowledge transfer')
cmd:option('-T', 20, 'Temperature for knowledge transfer')
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

-- Specialist creation
print(c.blue '==>' ..' creating specialists')
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
print(c.blue '==>' ..' loading data')
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

print(c.blue'==>' ..' setting criterion')
if opt.gpu == 'true' then
  criterion = DarkKnowledgeCriterion(opt.T, opt.alpha):cuda()
else
  criterion = DarkKnowledgeCriterion(opt.T, opt.alpha)
end

print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function populate_labels(input_targets, curr_domain)
  --- Creates special labels for specialists
    local output = input_targets:clone():fill(#curr_domain + 1)
    for i, val in ipairs(curr_domain) do
        output[input_targets:eq(val)] = i
    end
    return output
end


function populate_scores(input_scores, curr_domain, method)
  -- Creates special scores for specialists
  local raw_scores = input_scores:clone()
  output_scores = raw_scores:clone()
  output_scores:resize(raw_scores:size(1), #curr_domain + 1)
  output_scores:fill(0)
  for i, val in ipairs(curr_domain) do
    output_scores[{{}, i}] = raw_scores[{{}, val}]
    if method == 'max' then
      raw_scores[{{}, val}] = - math.huge
    else -- method == 'sum'
      raw_scores[{{}, val}] = 0.
    end
  end
  output_scores[{{}, #curr_domain}] = raw_scores:max(2)
  return output_scores
end


function train()
  -- Swith to train mode (flips, dropout, normalization)
  model:training()
  epoch = epoch or 1

  -- Drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then 
    optimState.learningRate = optimState.learningRate / 2 
  end
  
  print(c.blue '==>'.." online epoch # " .. 
    epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  targets = {}
  if opt.gpu == 'true' then
    targets.labels = torch.CudaTensor(opt.batchSize)
    targets.scores = torch.CudaTensor(opt.batchSize, num_class_specialist)
    temp = torch.CudaTensor(opt.batchSize)
  else
    targets.labels = torch.FloatTensor(opt.batchSize)
    targets.scores = torch.FloatTensor(opt.batchSize, num_class_specialist)
    temp = torch.FloatTensor(opt.batchSize)
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
    temp:copy(provider.trainData.label:index(1,v))
    targets.labels:copy(populate_labels(temp, domain))
    targets.scores:copy(populate_scores(provider.trainData.scores:index(1,v), 
      domain, 'max'))

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
  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1,provider.valData.data:size(1),bs do
    local outputs = model:forward(provider.valData.data:narrow(1,i,bs))
    local labels = populate_labels(provider.valData.label:narrow(1,i,bs),domain)
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
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- Save model every 'checkpoint' epochs
  if epoch % opt.checkpoint == 0 then
    local filename = paths.concat(opt.save, 'sp' .. opt.index .. 'ep '.. epoch .. '.net')
    print(c.blue '==>' .. 'saving model to '.. filename)
    torch.save(filename, model:get(3):clearState())
  end

  confusion:zero()
end

--> Actual training script
for i=1,opt.max_epoch do
  train()
  test()
end
