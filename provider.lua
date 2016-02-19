require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local trsize = 40000
  local vlsize = 10000
  local tesize = 10000

  -- download dataset
  if not paths.dirp('dataset') then
     local www = 'http://web.stanford.edu/~naromano/dataset/'
     local tar = paths.basename(www)
     os.execute('wget ' .. www .. '-r')
  end

  -- load dataset
  self.trainData = {
     data = torch.Tensor(40000, 3, 32, 32),
     label = torch.Tensor(40000),
     labelCoarse = torch.Tensor(40000),
     size = function() return trsize end
  }
  local trainData = self.trainData
  trainData = torch.load('dataset/cifar100-train.t7', 'ascii')
  trainData.label = trainData.label + 1
    
  self.valData = {
     data = torch.Tensor(40000, 3, 32, 32),
     label = torch.Tensor(40000),
     labelCoarse = torch.Tensor(40000),
     size = function() return trsize end
  }
  local valData = self.valData
  valData = torch.load('dataset/cifar100-test.t7', 'ascii')
  valData.label = valData.label + 1
    
  self.testData = {
     data = torch.Tensor(40000, 3, 32, 32),
     label = torch.Tensor(40000),
     labelCoarse = torch.Tensor(40000),
     size = function() return trsize end
  }
  local testData = self.testData
  testData = torch.load('dataset/cifar100-val.t7', 'ascii')
  testData.label = testData.label + 1

  -- resize dataset (if using small version)
  trainData.data = trainData.data[{ {1,trsize} }]
  trainData.labels = trainData.labels[{ {1,trsize} }]

  testData.data = testData.data[{ {1,tesize} }]
  testData.labels = testData.labels[{ {1,tesize} }]
end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.trainData
  local valData = self.valData
  local testData = self.testData

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v


  -- preprocess testSet
  for i = 1,valData:size() do
    xlua.progress(i, testData:size())
     -- rgb -> yuv
     local rgb = valData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     valData.data[i] = yuv
  end
  -- normalize u globally:
  valData.data:select(2,2):add(-mean_u)
  valData.data:select(2,2):div(std_u)
  -- normalize v globally:
  valData.data:select(2,3):add(-mean_v)
  valData.data:select(2,3):div(std_v)
    

  -- preprocess testSet
  for i = 1,testData:size() do
    xlua.progress(i, testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end
  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)
end