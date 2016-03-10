--[[ Specialist models architecture.

For now, it uses a very shallow version of VGG, with no BN. ]]--

require 'nn'

-- Create the model
local spec = nn.Sequential()

-- Building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  spec:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  -- spec:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
  spec:add(nn.ReLU(true))
  return spec
end

-- Use 'ceil' max pooling
local MaxPooling = nn.SpatialMaxPooling

-- VGG Architecture
ConvBNReLU(3,32)
spec:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(32,32)
spec:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(32,64) -- :add(nn.Dropout(0.4))
spec:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(64,64) -- :add(nn.Dropout(0.4))
spec:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(64,128) -- :add(nn.Dropout(0.4))
spec:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(128,128) -- :add(nn.Dropout(0.4))
spec:add(nn.View(128))
spec:add(nn.Linear(128,128))
spec:add(nn.BatchNormalization(128))
spec:add(nn.ReLU(true))
spec:add(nn.Linear(128, num_class_specialist))

-- Initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0, math.sqrt(2/n))
      v.bias:zero()
    end
  end
  init'nn.SpatialConvolution'
end

MSRinit(spec)
return spec
