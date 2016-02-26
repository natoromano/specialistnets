local DarkKnowledgeCriterion, parent = torch.class('DarkKnowledgeCriterion', 'nn.Criterion')

function DarkKnowledgeCriterion:__init(alpha, temp)
    -- alpha: weighted sum of the soft target loss and hard target loss
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
    -- target: at table where target.labels = labels and target.scores = raw scores of master
    local soft_traget = self.sm:forward(target.scores / self.temp)
    local log_probs = self.lsm:forward(input / self.temp)
    
    local output = (1 - self.alpha) * self.ce_crit:forward(input, target.labels)
    output = output + self.alpha * self.kl_crit:forward(log_probs, soft_target)
    return output
end

function DarkKnowledgeCriterion:updateGradInput(input, target)
    local soft_traget = self.sm:forward(target.scores / self.temp)
    local log_probs = self.lsm:forward(input / self.temp)
    local grad_ce = (1 - self.alpha) * self.ce_crit:backward(input, target.labels)
    local grad_kl = self.alpha * self.kl_crit:backward(log_probs, soft_target)
    grad_kl = grad_kl * self.temp * self.lsm:backward(input / self.temp)
    -- grad_kl was multiplied by T^2 as suggested by Hinton et al. 
    -- (Distilling the Knowledge in a Neural Network)
    return grad_ce + grad_kl
end