require 'torch'

math.randomseed(1234)

local c100 = torch.load('cifar100-trainFull.t7')
local nVal = 10000
local nSample = 50000
local nTrain = nSample - nVal

print(c100)
local mask = {}
local outTrain = {}
local outVal = {}

function contains(t, e)
  for i = 1,#t do
    if t[i] == e then return true end
  end
  return false
end


while #mask < nVal do
    j = math.random(1, nSample)
    if not contains(mask,j) then
        mask[#mask+1] = j
    end
end

iT = 1
iV = 1
dataT = torch.ByteTensor(nTrain, 3, 32, 32)
dataV = torch.ByteTensor(nVal, 3, 32, 32)
labelT = torch.ByteTensor(nTrain)
labelV = torch.ByteTensor(nVal)
coarseLabelT = torch.ByteTensor(nTrain)
coarseLabelV = torch.ByteTensor(nVal)
for i = 1,nSample do
    if contains(mask,i) then
        dataV[iV] = c100.data[{{i},{},{},{}}]
        labelV[iV] = c100.label[i]
        coarseLabelV[iV] = c100.labelCoarse[i]
        iV = iV + 1
    else
        dataT[iT] = c100.data[{{i},{},{},{}}]
        labelT[iT] = c100.label[i]
        coarseLabelT[iT] = c100.labelCoarse[i]
        iT = iT + 1
    end
end

outTrain.data = dataT
outTrain.label = labelT
outTrain.labelCoarse = coarseLabelT

outVal.data = dataV
outVal.label = labelV
outVal.labelCoarse = coarseLabelV

print(outTrain)
print(outVal)

torch.save('cifar100-train.t7', outTrain)
torch.save('cifar100-val.t7', outVal)

    
    
    