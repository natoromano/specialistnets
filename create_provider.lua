--[[ Creates provider for specialist nets.

This code is widely inspired by Sergey Zagoruyko, 
cf https://github.com/szagoruyko/cifar.torch ]]--

require 'xlua'
require 'optim'
require 'nn'
require 'provider.lua'
local c = require 'trepl.colorize'

-- Parameters
cmd = torch.CmdLine()
cmd:text('Create provider')
cmd:text()
cmd:text('Options')
cmd:option('-target', 'specialists', 'Target should be specialists or master')
cmd:option('-path', 'specialist_provider.t7', 'Path to save the provider')
cmd:option('-model', 'master/model.net', 'Path to the master model')
cmd:option('-data', 'master/master_provider.t7', 'Path to master provider')
cmd:option('-backend', 'nn')
cmd:text()

-- Parse input params
local opt = cmd:parse(arg)

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


function compute_scores(model, inputData, dim_output)
	--[[ Takes a trained model and a data object and returns the model's raw 
	scores on the data
	inputData must have a field .data, and a :size() method. ]]--
	local scores = torch.FloatTensor(inputData:size(), dim_output):zero()
	print scores:size()
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


if opt.target == 'master' then
	provider = Provider()
	provider:normalize()
	torch.save(path, provider)
end

if opt.target == 'specialists' then
	model = torch.load(opt.model)
	m_provider = torch.load(opt.data)
	scores = {}
	print(c.blue '==>'.."computing training scores...")
	scores.train = compute_scores(model, m_provider.trainData, 100)
	print(c.blue '==>'.."computing validation scores...")
	scores.val = compute_scores(model, m_provider.valData, 100)
	print(c.blue '==>'.."computing test scores...")
	scores.test = compute_scores(model, m_provider.testData, 100)
	torch.save('test.t7', scores)
	-- provider = Provider(scores)
	-- provider:normalize()
	-- torch.save(opt.path, provider)
end
