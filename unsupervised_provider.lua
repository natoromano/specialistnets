--[[ Provider to load and pre-process CIFAR-100 data.

This code is inspired by Sergey Zagoruyko, 
cf https://github.com/szagoruyko/cifar.torch ]]--

-- Imports
require 'nn'
require 'image'
require 'xlua'
torch.setdefaulttensortype('torch.FloatTensor')

-- Provider class
local UProvider = torch.class('UProvider')

function UProvider:__init(size, normalization, scores, domain)
  self.normalization = normalization
  if domain == nil then  
    -- Load dataset
    self.trainData = torch.load('./data/unsupervised.t7')
    self.trainData.data = self.trainData.data[{ {1, size} }]
    self.trainData.data = self.trainData.data:type('torch.FloatTensor')
    self.trainData.size = function() return self.trainData.data:size(1) end
    if scores ~= nil then
      self.trainData.scores = scores
    end
    
  else
    self.trainData = torch.load('./data/unsupervised.t7')
    self.trainData.data = self.trainData.data[{ {1, size} }]
    self.trainData.data = self.trainData.data:type('torch.FloatTensor')
    self.trainData.scores = populate_scores(scores, domain)
    self.trainData.size = function() return self.trainData.data:size(1) end
  end
end


function populate_scores(input_scores, curr_domain, method)
  -- Creates special scores for specialists
  method = method or 'max'
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


function UProvider:normalize()
  -- Thanks to Sergey Zagoruyko, cf https://github.com/szagoruyko/cifar.torch
  local trainData = self.trainData

  print 'Preprocessing data (color space + normalization)'
  collectgarbage()

  -- Preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1,
    image.gaussian1D(7))
  -- Preprocess set
  for i = 1, trainData:size() do
    xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- Normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- Normalize u globally:
  trainData.data:select(2,2):add(-self.normalization.mean_u)
  trainData.data:select(2,2):div(self.normalization.std_u)
  -- Normalize v globally:
  trainData.data:select(2,3):add(-self.normalization.mean_v)
  trainData.data:select(2,3):div(self.normalization.std_v)
end
