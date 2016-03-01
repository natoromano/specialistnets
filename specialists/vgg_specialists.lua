--[[ Specialist models architecture.

For now, it uses a shallower version of VGG, with no BN. ]]--

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
ConvBNReLU(32,32)
spec:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(32,64)
ConvBNReLU(64,64)
spec:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(64,128)
ConvBNReLU(128,128)
ConvBNReLU(128,128)
spec:add(MaxPooling(2,2,2,2):ceil())
spec:add(nn.View(128*4*4))
spec:add(nn.Linear(128*4*4,256))
spec:add(nn.BatchNormalization(256))
spec:add(nn.ReLU(true))
spec:add(nn.Linear(256,num_class_specialist))

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
