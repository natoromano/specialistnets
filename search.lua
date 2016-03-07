local log_file = io.open('search.log','w')
for i=1,10 do
  lr = 5*10^(torch.uniform(1,2.5))
  Temp = 1*10^(torch.uniform(0,2))
  str = 'th train_compressed.lua -T '.. Temp..' -learningRate '..lr..' -alpha 0.999 -epoch_step 20 -index '.. i
  log_file:write(str .. '\n')
  os.execute(str)
end
log_file:close()

