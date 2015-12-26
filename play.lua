require 'torch'

player1 = require 'naive_player'
player2 = require 'td_learner'

win1 = 0
win2 = 0
draw = 0

for i = 1,10000 do
	
	--take action
	act1 = player1.action()
	act2 = player2.action()
	
	--get reward
	if act1 == act2 then
		draw = draw + 1
		player1.reward(0)
		player2.reward(0)
	elseif (act1-act2)%3 == 1 then
		win1 = win1 + 1
		player1.reward(1)
		player2.reward(-1)
	else
		win2 = win2 + 1
		player1.reward(-1)
		player2.reward(1)
	end
	
	--get state
	player1.observe(act1, act2)
	player2.observe(act2, act1)
	
	
	print (act1, act2, 'player1_win', win1, 'player2_win', win2, 'draw', draw)
	
end