--[[ Custom criterion for Dark Knowledge Transfer

cf Hinton et al. - Distilling the Knowledge in a Neural Network ]]--

local DarkKnowledgeCriterion, parent = torch.class('DarkKnowledgeCriterion',
                                                   'nn.Criterion')

function DarkKnowledgeCriterion:__init(alpha, temp)
    -- alpha: soft cross-entropy weight
    -- temp: temperature
    parent.__init(self)
    self.temp = temp or 1.0
    self.alpha = alpha or 0.9
    self.supervised = (self.alpha < 1.0)
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
    if self.supervised then
        self.output = self.ce_crit:forward(input, target.labels)*(1-self.alpha)
        self.output = self.output +
            self.kl_crit:forward(log_probs, soft_target) * self.alpha
    else
        self.output = self.kl_crit:forward(log_probs, soft_target)
    end
    return self.output
end

function DarkKnowledgeCriterion:updateGradInput(input, target)
    self.mask = target.labels:eq(0)
    local soft_target = self.sm:forward(target.scores:div(self.temp))
    local log_probs = self.lsm:forward(input:div(self.temp))
    if self.supervised then
        local grad_ce = self.ce_crit:backward(input, 
                                    target.labels):mul(1 - self.alpha)
        local grad_kl = self.kl_crit:backward(log_probs, 
                                    soft_target):mul(self.alpha)
        grad_kl = self.lsm:backward(input:div(self.temp),grad_kl):mul(self.temp)
        -- grad_kl is multiplied by T^2 as recommended by Hinton et al. 
        self.gradInput = grad_ce + grad_kl
    else
        local grad_kl = self.kl_crit:backward(log_probs, soft_target)
        grad_kl = self.lsm:backward(input:div(self.temp),grad_kl):mul(self.temp)
        -- grad_kl is multiplied by T^2 as recommended by Hinton et al. 
        self.gradInput = grad_kl        
    return self.gradInput
end
