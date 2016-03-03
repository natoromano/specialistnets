--[[ Provider to load and pre-process CIFAR-100 data.

This code is inspired by Sergey Zagoruyko, 
cf https://github.com/szagoruyko/cifar.torch ]]--

-- Imports
require 'nn'
require 'image'
require 'xlua'
torch.setdefaulttensortype('torch.FloatTensor')

-- Provider class
local Provider = torch.class 'Provider'

function Provider:__init(scores, domain)
  local trsize = 40000
  local vlsize = 10000
  local tesize = 10000

  if domain == nil then  
    -- Load dataset
    self.trainData = torch.load('./data/cifar100-train.t7')
    self.trainData.data = self.trainData.data:type('torch.FloatTensor')
    self.trainData.label = self.trainData.label + 1
    self.trainData.size = function() return trsize end
    if scores ~= nil then
      self.trainData.scores = scores.train
    end
    
    self.valData = torch.load('./data/cifar100-val.t7')
    self.valData.data = self.valData.data:type('torch.FloatTensor')
    self.valData.label = self.valData.label + 1
    self.valData.size = function() return vlsize end
    if scores ~= nil then
      self.valData.scores = scores.val
    end
    
    self.testData = torch.load('./data/cifar100-test.t7')
    self.testData.data = self.testData.data:type('torch.FloatTensor')
    self.testData.label = self.testData.label + 1
    self.testData.size = function() return tesize end
    if scores ~= nil then
      self.testData.scores = scores.test
    end

    -- Resize dataset (if using small version)
    self.trainData.data = self.trainData.data[{ {1, trsize} }]
    self.trainData.label = self.trainData.label[{ {1, trsize} }]
    if scores ~= nil then
      self.trainData.scores = self.trainData.scores[{ {1, trsize} }]
    end

    self.valData.data = self.valData.data[{ {1, vlsize} }]
    self.valData.label = self.valData.label[{ {1, vlsize} }]
    if scores ~= nil then
      self.valData.scores = self.valData.scores[{ {1, vlsize} }]
    end
    
    self.testData.data = self.testData.data[{ {1, tesize} }]
    self.testData.label = self.testData.label[{ {1, tesize} }]
    if scores ~= nil then
      self.testData.scores = self.testData.scores[{ {1, tesize} }]
    end
    
  else
    --[[ Train Data: Select parts of the data such that it is composed of 50% 
    specialist labels and 50% dustbin labels]]
    local fullTrainData = torch.load('./data/cifar100-train.t7')
    self.trainData = {}
    local mask = torch.ByteTensor(trsize):fill(0)
    -- Create mask to choose the right data
    for i, val in ipairs(domain) do
      mask = mask + fullTrainData.label:eq(val)
    end
    local indices = torch.linspace(1,trsize,trsize):long()
    local N = mask:eq(0):sum() -- number of examples where mask == 0
    local M = mask:eq(1):sum() -- number of examples where mask > 1
    mask[mask:eq(0)] = torch.rand(N):lt(M/N)
    local selected = indices[mask:eq(1)]
    self.trainData.data = fullTrainData.data:index(1,selected):type('torch.FloatTensor')
    self.trainData.label = populate_labels(fullTrainData.label:index(1,selected), domain)
    self.trainData.scores = populate_scores(scores.train:index(1,selected), domain)
    self.trainData.size = function() return self.trainData.data:size(1) end
    
    --[[ Val Data: Select parts of the data such that it is composed of 50% 
    specialist labels and 50% dustbin labels]]
    local fullValData = torch.load('./data/cifar100-val.t7')
    self.valData = {}
    local mask = torch.ByteTensor(vlsize):fill(0)
    -- Create mask to choose the right data
    for i, val in ipairs(domain) do
      mask = mask + fullValData.label:eq(val)
    end
    local indices = torch.linspace(1,vlsize,vlsize):long()
    local N = mask:eq(0):sum() -- number of examples where mask == 0
    local M = mask:eq(1):sum() -- number of examples where mask > 1
    mask[mask:eq(0)] = torch.rand(N):lt(M/N)
    local selected = indices[mask:eq(1)]
    self.valData.data = fullValData.data:index(1,selected):type('torch.FloatTensor')
    self.valData.label = populate_labels(fullValData.label:index(1,selected), domain)
    self.valData.scores = populate_scores(scores.val:index(1,selected), domain)
    self.valData.size = function() return self.valData.data:size(1) end
    
    --[[ Test Data: Select parts of the data such that it is composed of 50% 
    specialist labels and 50% dustbin labels]]
    local fullTestData = torch.load('./data/cifar100-test.t7')
    self.testData = {}
    local mask = torch.ByteTensor(tesize):fill(0)
    -- Create mask to choose the right data
    for i, val in ipairs(domain) do
      mask = mask + fullTestData.label:eq(val)
    end
    local indices = torch.linspace(1,tesize,tesize):long()
    local N = mask:eq(0):sum() -- number of examples where mask == 0
    local M = mask:eq(1):sum() -- number of examples where mask > 1
    mask[mask:eq(0)] = torch.rand(N):lt(M/N)
    local selected = indices[mask:eq(1)]
    self.testData.data = fullTestData.data:index(1,selected):type('torch.FloatTensor')
    self.testData.label = populate_labels(fullTestData.label:index(1,selected), domain)
    self.testData.scores = populate_scores(scores.test:index(1,selected), domain)
    self.testData.size = function() return self.testData.data:size(1) end
  end
end

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
  output_scores[{{}, #curr_domain + 1}] = raw_scores:max(2)
  return output_scores
end


function Provider:normalize()
  -- Thanks to Sergey Zagoruyko, cf https://github.com/szagoruyko/cifar.torch
  local trainData = self.trainData
  local valData = self.valData
  local testData = self.testData

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- Preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, 
                                          image.gaussian1D(7))
  for i = 1, trainData:size() do
     xlua.progress(i, trainData:size())
     -- RGB -> YUV
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- Normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- Normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)
  -- Save means
  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v

  -- Preprocess valSet
  for i = 1, valData:size() do
    xlua.progress(i, valData:size())
     -- rgb -> yuv
     local rgb = valData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- Normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     valData.data[i] = yuv
  end
  -- Normalize u globally:
  valData.data:select(2,2):add(-mean_u)
  valData.data:select(2,2):div(std_u)
  -- Normalize v globally:
  valData.data:select(2,3):add(-mean_v)
  valData.data:select(2,3):div(std_v)

  -- Preprocess testSet
  for i = 1, testData:size() do
    xlua.progress(i, testData:size())
     -- RGB -> YUV
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- Normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end
  -- Normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- Normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)
end
