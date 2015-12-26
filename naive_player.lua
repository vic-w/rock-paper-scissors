print 'I am naive player'

local naive_player = {}

local last = 1;

function naive_player.action()
	last = (last%3 + 1)
	return last
end

function naive_player.observe(my_action, op_action)
end

function naive_player.reward(r)
end
	
return naive_player