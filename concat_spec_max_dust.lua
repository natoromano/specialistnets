--[[ To predict:
choose the specialist which has the lowest dustbin probability
run specialist_scores.lua before this
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

-- Import necessary modules
if opt.backend == 'cudnn' then
	require 'cunn'
	require 'cudnn'
	require 'cutorch'
	cudnn.fastest, cudnn.benchmark = true, true
end

if opt.backpend == 'cunn' then
	require 'cunn'
	require 'cutorch'
end

-- Data loading
print(c.blue '==>' ..' loading data')
if string.find(opt.data, 'mnt') then
    os.execute('sudo chmod 777 ' .. opt.data)
end

domains = torch.load(opt.domains)

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


print(c.blue'==>' ..' setting criterion')
sm = nn.SoftMax()

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

function create_prob_distribution(inputs, domains)
  -- initialize outputs
  local bs = inputs:size(1)
  local outputs = torch.FloatTensor(bs,100)
  local id = 0
  local min_dustbin_prob = 1.1
  --[[ for every specialist check who has the smallest dustbin prob and copy its
  probability distribution ]]--
  for j = 1, bs do
    id = 0
    for i, domain in pairs(domains) do
      local dim = #domain + 1
      probs = sm:forward(inputs[{{j},{id+1, id+dim}}])
      min_dustbin_prob = 1.1
      if (probs[#probs] < min_dustbin_prob) then
        min_dustbin_prob = probs[#probs]
        outputs[{{j},{}}]:zero()
	for k, class in pairs(domain) do
	  outputs[{j,class}] = probs[{1,k}]
	end
      end
      id = id + dim
    end
  end
  return outputs
end 
  

local tic = torch.tic()
-- Iterate over batches
for t,v in ipairs(indices) do
  xlua.progress(t, #indices)

  local inputs = provider.trainData.data:index(1,v)
  local outputs = create_prob_distribution(inputs, domains)
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
  local inputs = provider.valData.data:narrow(1,i,bs)
  local outputs = create_prob_distribution(inputs, domains)
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
  file:write'</table><pre>\n'
  file:write(tostring(confusion)..'\n')
  file:write(tostring(model)..'\n')
  file:write'</pre></body></html>'
  file:close()
end

confusion:zero()

