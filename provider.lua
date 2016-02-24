-- This code was widely inspired by Sergey Zagoruyko
-- cf https://github.com/szagoruyko/cifar.torch

require 'nn'
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local trsize = 40000
  local vlsize = 10000
  local tesize = 10000

  -- Load dataset
  self.trainData = torch.load('./data/cifar100-train.t7')
  self.trainData.data = self.trainData.data:type('torch.FloatTensor')
  self.trainData.label = self.trainData.label + 1
  self.trainData.size = function() return trsize end
    
  self.valData = torch.load('./data/cifar100-val.t7')
  self.valData.data = self.valData.data:type('torch.FloatTensor')
  self.valData.label = self.valData.label + 1
  self.valData.size = function() return vlsize end
    
  self.testData = torch.load('./data/cifar100-test.t7')
  self.testData.data = self.testData.data:type('torch.FloatTensor')
  self.testData.label = self.testData.label + 1
  self.testData.size = function() return tesize end

  -- Resize dataset (if using small version)
  self.trainData.data = self.trainData.data[{ {1, trsize} }]
  self.trainData.label = self.trainData.label[{ {1, trsize} }]

  self.valData.data = self.valData.data[{ {1, vlsize} }]
  self.valData.label = self.valData.label[{ {1, vlsize} }]
    
  self.testData.data = self.testData.data[{ {1, tesize} }]
  self.testData.label = self.testData.label[{ {1, tesize} }]
end


function Provider:normalize()
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
     -- rgb -> yuv
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
     -- rgb -> yuv
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
