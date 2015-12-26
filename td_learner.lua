require 'torch'
require 'nn'
require 'rnn'

print 'I am TD learner'

local td_learner = {}

local last = 1;
local last_my_action = 0
local last_op_action = 0

local hiddenSize = 100
local rho = 10

local r = nn.Recurrent(
   hiddenSize, nn.Linear(6, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

local model = nn.Sequential()
model:add(r)
model:add(nn.Linear(hiddenSize, 3))
model:add(nn.SoftMax())
criterion = nn.MSECriterion()

function td_learner.action()
	return 2
end

function td_learner.observe(my_action, op_action)
	last_my_action = my_action
	last_op_action = op_action
end

function td_learner.reward(r)
end
	
return td_learner