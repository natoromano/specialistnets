--[[ Custom criterion for Dark Knowledge Transfer

cf Hinton et al. - Distilling the Knowledge in a Neural Network ]]--

local DarkKnowledgeCriterion, parent = torch.class('DarkKnowledgeCriterion',
                                                   'nn.Criterion')

function DarkKnowledgeCriterion:__init(alpha, temp, soft_loss, verbose)
    -- alpha: soft cross-entropy weight
    -- temp: temperature
    -- soft_loss: 'KL', 'MSE' or 'L1'
    parent.__init(self)
    self.temp = temp or 1.0
    self.alpha = alpha or 0.9
    self.soft_loss = soft_loss or 'KL'
    self.supervised = (self.alpha < 1.0)
    self.verbose = verbose or false
    self.sm = nn.SoftMax()
    self.lsm = nn.LogSoftMax()
    self.ce_crit = nn.CrossEntropyCriterion()
    if self.soft_loss == 'KL' then
        self.kl_crit = nn.DistKLDivCriterion()
    elseif self.soft_loss == 'L2' then
        self.mse_crit = nn.MSECriterion()
        self.mse_crit.sizeAverage = false
    else
        error('invalid input as soft_loss')
    end
end

function DarkKnowledgeCriterion:updateOutput(input, target)
    -- input: raw scores from the model
    -- target.labels = ground truth labels
    -- target.scores = raw scores from the master
    local soft_target = self.sm:forward(target.scores / self.temp):clone()
    if self.soft_loss == 'KL' then
      local log_probs = self.lsm:forward(input / self.temp)
      if self.supervised then
          self.output = self.ce_crit:forward(input, target.labels) * (1-self.alpha)
          if self.verbose then
            local str = string.format('CE/KL loss: %1.0e/%1.0e', self.output,
                           self.kl_crit:forward(log_probs, soft_target) * self.alpha)
            print(str)
          end 
          self.output = self.output +
              self.kl_crit:forward(log_probs, soft_target) * self.alpha
      else
          self.output = self.kl_crit:forward(log_probs, soft_target)
      end
    else
      local probs = self.sm:forward(input / self.temp)
      if self.supervised then
          self.output = self.ce_crit:forward(input, target.labels) * (1-self.alpha)
          -- local str = string.format('CE/MSE loss: %1.0e/%1.0e', self.output,
          --                   self.mse_crit:forward(probs, soft_target) * self.alpha) 
          self.output = self.output +
              self.mse_crit:forward(probs, soft_target) * self.alpha
          -- print(str)
      else
          self.output = self.mse_crit:forward(probs, soft_target)
      end
    end
    return self.output
end

function DarkKnowledgeCriterion:updateGradInput(input, target)
    self.mask = target.labels:eq(0)
    local soft_target = self.sm:forward(target.scores:div(self.temp)):clone()
    if self.soft_loss == 'KL' then
      local log_probs = self.lsm:forward(input / self.temp)
      if self.supervised then
          local grad_ce = self.ce_crit:backward(input, 
                                      target.labels) * (1 - self.alpha)
          local grad_kl = self.kl_crit:backward(log_probs, 
                                      soft_target) * (self.alpha)
          grad_kl = self.lsm:backward(input:div(self.temp),grad_kl) * (self.temp)
          --grad_kl is multiplied by T^2 as recommended by Hinton et al. 
          self.gradInput = grad_ce + grad_kl
          if self.verbose then
            local str = string.format('CE/KL grad:     %1.0e/%1.0e', grad_ce:norm(), grad_kl:norm())
            print(str)
          end
      else
          local grad_kl = self.kl_crit:backward(log_probs, soft_target)
          grad_kl = self.lsm:backward(input:div(self.temp),grad_kl) * (self.temp)
          -- grad_kl is multiplied by T^2 as recommended by Hinton et al. 
          self.gradInput = grad_kl        
      end
    else
      local probs = self.sm:forward(input:div(self.temp))
      if self.supervised then
          local grad_ce = self.ce_crit:backward(input, 
                                      target.labels) * (1 - self.alpha)
          local grad_mse = self.mse_crit:backward(probs, 
                                      soft_target) * (self.alpha)
          grad_mse = self.sm:backward(input:div(self.temp),grad_mse) * (self.temp)
          -- grad_kl is multiplied by T^2 as recommended by Hinton et al. 
          self.gradInput = grad_ce + grad_mse
          -- local str = string.format('CE/MSE grad:     %1.0e/%1.0e', grad_ce:norm(), grad_mse:norm())
          -- print(str)
      else
          local grad_mse = self.mse_crit:backward(probs, soft_target)
          grad_mse = self.sm:backward(input:div(self.temp),grad_mse) * (self.temp)
          -- grad_kl is multiplied by T^2 as recommended by Hinton et al. 
          self.gradInput = grad_mse        
      end
    end
    return self.gradInput
end
