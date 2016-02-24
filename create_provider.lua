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
cmd:option('-data', 'master/mater_provider.t7', 'Path to master provider')
cmd:option('-gpu', false)
cmd:text()


function compute_scores(model, inputData, dim_output)
	--[[ Takes a trained model and a data object and returns the model's raw 
	scores on the data
	inputData must have a field .data, and a :size() method. ]]--
	scores = torch.FloatTensor(inputData:size(), dim_output):zero()
	local bs = 125  -- batch size for forward pass
	for i = 1, inputData.data.size(1), bs do
		if gpu == true then
			data = inputData.data:narrow(1, i, bs):cuda()
		else
			data = inputData.data:narrow(1, i, bs)
		local outputs = model:forward(data):float()
		scores[{ {i*bs, (i+1)*bs-1} }] = outputs
	end
	return scores
end


-- Parse input params
local opt = cmd:parse(arg)

if opt.target == 'master' then
	provider = Provider()
	provider:normalize()
	torch.save(path, provider)
end

if opt.target == 'specialists' then
	model = torch.load(opt.model)
	m_provider = torch.load(opt.data)
	scores = {}
	scores.train = compute_scores(model, m_provider.trainData, 100)
	scores.val = compute_scores(model, m_provider.valData, 100)
	scores.test = compute_scores(model, m_provider.testData, 100)
	provider = Provider(scores=scores)
	provider:normalize()
	torch.save(path, provider)
end
