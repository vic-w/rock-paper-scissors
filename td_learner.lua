require 'torch'
require 'nn'
require 'rnn'
require 'dpnn'

print 'I am TD learner'

local td_learner = {}

local last = 1;
local my_reward = 0;
local last_my_action = 1
local last_op_action = 1

local hiddenSize = 100
local rho = 10

local r = nn.Recurrent(
   hiddenSize, nn.Linear(6, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

predictor = nn.Sequential()
--predictor:add(r)
l = nn.Linear(6, 3)
l.weight:fill(0)
predictor:add(l)
predictor:add(nn.SoftMax())
predictor:add(nn.ReinforceCategorical())

bias_learner = nn.Sequential()
bias_learner:add(nn.Constant(1,1))
c = nn.Add(1)
c.bias = -1
bias_learner:add(nn.Add(1))

model = nn.ConcatTable()
model:add(predictor)
model:add(bias_learner)

criterion = nn.VRClassReward(model)

function td_learner.action()
    input = torch.zeros(1,6)
    input[1][last_my_action] = 1
    input[1][last_op_action + 3] = 1
    --print(input)
    --print(l.weight)
    predict = model:forward(input)
    --print(predict[1])
    y,i = predict[1]:max(2)
	return i[1][1]
end

function td_learner.observe(my_action, op_action)
    input = torch.zeros(1,6)
    input[1][last_my_action] = 1
    input[1][last_op_action + 3] = 1
    target = torch.Tensor{my_action}
    --print(input)
    predict = model:forward(input)
    --print(predict)
    --print(c.bias)
    criterion:forward(predict, target)
    gradient = criterion:backward(predict, target)
    --print(gradient)
    model:backward(input, gradient)
    model:updateParameters(0.0001* my_reward)
        
	last_my_action = my_action
	last_op_action = op_action
end

function td_learner.reward(result)
    my_reward = result;
end

return td_learner
