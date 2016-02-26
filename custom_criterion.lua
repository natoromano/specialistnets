--[[ Custom criterion for Dark Knowledge Transfer

cf Hinton et al. - Distilling the Knowledge in a Neural Network ]]--

local DarkKnowledgeCriterion, parent = torch.class('DarkKnowledgeCriterion', 'nn.Criterion')

function DarkKnowledgeCriterion:__init(alpha, temp)
    -- alpha: soft cross-entropy weight
    -- temp: temperature
    parent.__init(self)
    self.temp = temp or 1.0
    self.alpha = alpha or 0.9
    self.sm = nn.SoftMax()
    self.lsm = nn.LogSoftMax()
    self.ce_crit = nn.CrossEntropyCriterion()
    self.kl_crit = nn.DistKLDivCriterion()
end

function DarkKnowledgeCriterion:updateOutput(input, target)
    -- input: raw scores from the model
    -- target.labels = ground truth labels
    -- target.scores = raw scores from the master
    local soft_target = self.sm:forward(target.scores:div(self.temp))
    local log_probs = self.lsm:forward(input:div(self.temp))
    local output = self.ce_crit:forward(input, target.labels) * (1 - self.alpha)
    output = output + self.kl_crit:forward(log_probs, soft_target) * self.alpha
    return output
end

function DarkKnowledgeCriterion:updateGradInput(input, target)
    local soft_target = self.sm:forward(target.scores:div(self.temp))
    local log_probs = self.lsm:forward(input:div(self.temp))
    local grad_ce = self.ce_crit:backward(input, target.labels):mul(1 - self.alpha)
    local grad_kl = self.kl_crit:backward(log_probs, soft_target):mul(self.alpha)
    grad_kl = self.lsm:backward(input:div(self.temp), grad_kl):mul(self.temp)
    -- grad_kl was multiplied by T^2 as suggested by Hinton et al. 
    return grad_ce + grad_kl
end
