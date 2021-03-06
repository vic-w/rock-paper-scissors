require 'torch'
require 'nn'
require 'rnn'
require 'dpnn'

print 'I am TD learner'

local td_learner = {}

local last = 1;
local my_reward = 0;
local last_my_action = 2
local last_op_action = 2

local hiddenSize = 100
local rho = 10

local r = nn.Recurrent(
   hiddenSize, nn.Linear(6, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

predictor = nn.Sequential()
--predictor:add(r)
local l = nn.Linear(3, 3)
l.weight:fill(0.1)
l.bias:fill(1)
local w, dl_dw = l:getParameters()
predictor:add(l)
--predictor:add(nn.Sigmoid())
predictor:add(nn.SoftMax())
predictor:add(nn.ReinforceCategorical())

bias_learner = nn.Sequential()
bias_learner:add(nn.Constant(1,1))
local c = nn.Add(1)
c.bias:fill(-1)
bias_learner:add(c)

model = nn.ConcatTable()
model:add(predictor)
model:add(bias_learner)

criterion = nn.VRClassReward(model)

function td_learner.action()
    --print("Action")
    if torch.uniform() < 0.1 then 
        --print('random')
        dice = torch.uniform()
        if dice < 1/3 then
            return 1
        elseif dice < 2/3 then
            return 2
        else
            return 3
        end
    else
        input = torch.zeros(1,3)
        --input[1][last_my_action] = 1
        input[1][last_op_action] = 1
        --print(input)
        --print(l.weight)
        l.bias:fill(1)
        --print(l.bias)
        predict = model:forward(input)
        --print('predict', predict[1])
        y,i = predict[1]:max(2)
	    return i[1][1]
	end
end

function td_learner.observe(my_action, op_action)
    --print('Observe')
    input = torch.zeros(1,3)
    --input[1][last_my_action] = 1
    input[1][last_op_action] = 1
    --print('input',input)
    target = torch.Tensor{op_action%3+1}
    --print('target', target)
    --print(input)
    predict = model:forward(input)
    --print(l.output)
    --print(predict)
    --print(c.bias)
    --print('bias = '..predict[2][{1,1}])
    criterion:forward(predict, target)
    gradient = criterion:backward(predict, target)
    --print(gradient)
    model:backward(input, gradient)
    model:updateParameters(0.1)
    
    c.bias:fill(-1)
        
	last_my_action = my_action
	last_op_action = op_action
end

function td_learner.reward(result)
    my_reward = result;
end

return td_learner
