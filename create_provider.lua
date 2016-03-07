--[[ Creates provider for master or specialist nets. ]]--

-- Imports
require 'xlua'
require 'nn'
dofile 'provider.lua'
dofile 'unsupervised_provider.lua'

-- Parameters
cmd = torch.CmdLine()
cmd:text('Create provider')
cmd:text()
cmd:text('Options')
cmd:option('-target', 'specialists', 'Target should be specialists, compressed  or master')
cmd:option('-method', 'supervised', 'Should be supervised or unsupervised')
cmd:option('-provider', 'master/master_provider.t7', 
	'Path to master (or specialist) provider for normalization constants')
cmd:option('-size', 20000, 'Size of unsupervised training set')
cmd:option('-path', '/mnt', 'Path to save the provider')
cmd:option('-scores', 'master/scores250.t7', 'Path to master scores')
cmd:option('-backend', 'cudnn')
cmd:option('-domains', 'specialists/new.t7')
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

if opt.method == 'supervised' then
	-- Master provider creation
	if opt.target == 'master' then
		provider = Provider()
		provider:normalize()
		-- Change permissions on temp mnt/ directory on AWS
		if string.find(opt.path, 'mnt') then
			os.execute('sudo chmod 777 ' .. opt.path)
		end
		torch.save(opt.path .. '/master_provider.t7', provider)
	end

	if opt.target == 'compressed' then
	  scores = torch.load(opt.scores)
		provider = Provider(scores)
		provider:normalize()
		-- Change permissions on temp mnt/ directory on AWS
		if string.find(opt.path, 'mnt') then
			os.execute('sudo chmod 777 ' .. opt.path)
		end
		torch.save(opt.path .. '/compressed_provider.t7', provider)
	end	

	-- Specialist provider creation (same but with scores)
	if opt.target == 'specialists' then
	  domains = torch.load(opt.domains)
		scores = torch.load(opt.scores)
		
		-- Change permissions on temp mnt/ directory on AWS
		if string.find(opt.path, 'mnt') then
			os.execute('sudo chmod 777 ' .. opt.path)
		end
		for i, domain in ipairs(domains) do
		  provider = Provider(scores, domain)
		  provider:normalize()
	    torch.save(opt.path .. '/specialist' .. i .. '_provider.t7', provider)
	  end
	end

else
	norm_provider = torch.load(opt.provider)
	normalization = {}
	normalization.mean_u = norm_provider.trainData.mean_u
	normalization.std_u = norm_provider.trainData.std_u
	normalization.mean_v = norm_provider.trainData.mean_v
	normalization.std_v = norm_provider.trainData.std_v

	-- Master provider creation
	if opt.target == 'master' then
		provider = UProvider(opt.size, normalization)
		provider:normalize()
		-- Change permissions on temp mnt/ directory on AWS
		if string.find(opt.path, 'mnt') then
			os.execute('sudo chmod 777 ' .. opt.path)
		end
		torch.save(opt.path .. '/master_uprovider.t7', provider)
	end

	-- Specialist provider creation (same but with scores)
	if opt.target == 'specialists' then
	    domains = torch.load(opt.domains)
		scores = torch.load(opt.scores)
		
		-- Change permissions on temp mnt/ directory on AWS
		if string.find(opt.path, 'mnt') then
			os.execute('sudo chmod 777 ' .. opt.path)
		end
		for i, domain in ipairs(domains) do
		  provider = UProvider(opt.size, normalization, scores, domain)
		  provider:normalize()
	      torch.save(opt.path .. '/specialist' ..i.. '_uprovider.t7', provider)
	  end
	end	

	if opt.target == 'compressed' then
	  	scores = torch.load(opt.scores)
		provider = UProvider(scores, normalization)
		provider:normalize()
		-- Change permissions on temp mnt/ directory on AWS
		if string.find(opt.path, 'mnt') then
			os.execute('sudo chmod 777 ' .. opt.path)
		end
		torch.save(opt.path .. '/compressed_uprovider.t7', provider)
	end
end
