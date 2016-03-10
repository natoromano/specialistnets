local log_file = io.open('search_specialist.log','w')
best_val = torch.FloatTensor(9):fill(0)
for i=1,10 do
  for j=1,9 do
    hard_strength = 1*10^(torch.uniform(-4,-1)
    alpha = 1 - hard_strength
    lr = 1*10^(torch.uniform((0,2) - 3) / hard_strength
    Temp = 1*10^(torch.uniform(0,1.5))
    str = 'th train_specialists.lua -T '.. Temp..' -learningRate '..lr..' -alpha ' .. alpha ..
            ' -max_epoch 130 -epoch_step 30 -checkpoint 130 -pretrained true' ..
            ' -m \' trying massive search\' -index ' .. j
    dofile(str)
    if (val_running_mean > best_val[j]) then
      print('found best for spec ' .. j .. ' at val: ' .. val_running_mean)
      best_val[j] = val_running_mean
      log_file:write('best with: ' .. val_running_mean .. ' ' .. str .. '\n')
      os.execute('cp specialist_logs/' ..model_name.. ' specialist_logs/best_spec_' ..j.. '.net')
      os.execute('cp specialist_logs/report' ..j.. '.html specialist_logs/best_report' ..j.. '.html') 
    else
      log_file:write(str .. '\n')
    end
  end
end
log_file:close()

