--[[ Given a master network, compute the raw scores on CIFAR-100. ]]--

-- Imports
require 'xlua'
require 'nn'
dofile 'provider.lua'
local c = require 'trepl.colorize'

-- Parameters
cmd = torch.CmdLine()
cmd:text('Create provider')
cmd:text()
cmd:text('Options')
cmd:option('-path', 'master/master_scores.t7', 'Path to save the scores')
cmd:option('-model', 'master/model.net', 'Path to the master model')
cmd:option('-data', '/mnt', 'Path to master provider')
cmd:option('-backend', 'cudnn')
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

-- Score computation function for a single data set
function compute_scores(model, inputData, dim_output)
	--[[ Takes a trained model and a data object and returns the model's raw 
	scores on the data
	inputData must have a field .data, and a :size() method. ]]--
	local scores = torch.FloatTensor(inputData:size(), dim_output):zero()
	local bs = 125  -- batch size for forward pass
	for i = 1, inputData.data:size(1), bs do
		if opt.backend == 'cudnn' or opt.backend == 'cunn' then
			data = inputData.data:narrow(1, i, bs):cuda()
		else
			data = inputData.data:narrow(1, i, bs)
		end
		local outputs = model:forward(data):float()
		scores[{{i, i-1}] = outputs
	end
	return scores
end

-- Load model
model = torch.load(opt.model)
-- load data
os.execute('sudo chmod 777 ' .. opt.data)
m_provider = torch.load(opt.data .. '/master_provider.t7')
-- Compute scores
scores = {}
print(c.blue '==>'.." computing training scores...")
scores.train = compute_scores(model, m_provider.trainData, 100)
print(c.blue '==>'.." computing validation scores...")
scores.val = compute_scores(model, m_provider.valData, 100)
print(c.blue '==>'.." computing test scores...")
scores.test = compute_scores(model, m_provider.testData, 100)
-- Save scores
torch.save(opt.path, scores)
