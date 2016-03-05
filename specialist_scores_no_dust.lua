--[[ Given specialist networks, compute the raw scores on CIFAR-100. ]]--

-- Imports
require 'xlua'
require 'nn'
dofile 'provider.lua'
local c = require 'trepl.colorize'

-- Parameters
cmd = torch.CmdLine()
cmd:text('Dump specialist scores')
cmd:text()
cmd:text('Options')
cmd:option('-path', 'specialists/specialist_scores_no_dust.t7', 'Path to save scores')
cmd:option('-models', '/mnt', 'Path to the specialist models')
cmd:option('-domains', 'specialists/new.t7', 'Path to domains')
cmd:option('-epochs', 150, 'Number of epochs specialists were trained with')
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
	-- local output = {}
	local scores = torch.FloatTensor(inputData:size(), dim_output):zero()
	local bs = 125  -- batch size for forward pass
	for i = 1, inputData.data:size(1), bs do
		if opt.backend == 'cudnn' or opt.backend == 'cunn' then
			data = inputData.data:narrow(1, i, bs):cuda()
		else
			data = inputData.data:narrow(1, i, bs)
		end
		local outputs = model:forward(data):float()
		scores[{{i, i+bs-1}}] = outputs
	end
	return scores
end

-- changing permissions on data file (for AWS use)
os.execute('sudo chmod 777 ' .. opt.data)

-- number of output classes
domains = torch.load(opt.domains)
local n_classes = 100

-- initialize score tensors
local train = torch.FloatTensor(40000, n_classes)
local val = torch.FloatTensor(10000, n_classes)
local test = torch.FloatTensor(10000, n_classes)

-- load data
provider = torch.load(opt.data .. '/master_provider.t7')

-- compute scores
local id = 0
for i, domain in pairs(domains) do
	print(c.blue '==>'.." computing scores for specialist" .. i .. "...")
	-- load model
	model = torch.load(opt.models .. '/sp' .. i .. 'ep' .. opt.epochs .. '.net')
	-- Compute scores
	local dim = #domain + 1
	sp_train = compute_scores(model, provider.trainData, dim)
	sp_val = compute_scores(model, provider.valData, dim)
	sp_test = compute_scores(model, provider.testData, dim)
	for j, class in pairs(domain) do
	  train[{{},{class}}] = sp_train[{{},{j}}]
	  val[{{},{class}}] = sp_val[{{},{j}}]
	  test[{{},{class}}] = sp_test[{{},{j}}]
	end
end

-- Save scores
function populate_scores(outputTable, setName, setScores)
	outputTable[setName] = {}
	outputTable[setName].data = setScores
	outputTable[setName].label = provider[setName].label
end

scores = {}
populate_scores(scores, 'trainData', train)
populate_scores(scores, 'valData', val)
populate_scores(scores, 'testData', test)
torch.save(opt.path, scores)
