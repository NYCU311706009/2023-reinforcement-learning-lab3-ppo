from ppo_agent_atari import AtariPPOAgent
import datetime
if __name__ == '__main__':
	current_time = datetime.datetime.now()
	current_year = current_time.year
	current_month = current_time.month
	current_day = current_time.day
	current_hour = current_time.hour
	current_minute = current_time.minute
	s = str(current_year) + str(current_month) + str(current_day) + str(current_hour) + str(current_minute)
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"update_sample_count": 10000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 128,
		"logdir": 'log/Enduro/' + s + '/',
		"update_ppo_epoch": 3,
		"learning_rate": 2.5e-4,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"env_id": 'ALE/Enduro-v5',
		# "env_id": 'ALE/MsPacman-v5',
		"eval_interval": 100,
		"eval_episode": 3,
	}
	agent = AtariPPOAgent(config)
	# pre_model = r'C:\Users\WANG\Desktop\強化學習專論\Lab03\ppo_agent_atari\log\Enduro\model_93895776_953.pth'
	pre_model = r'C:\Users\WANG\Desktop\強化學習專論\Lab03\ppo_agent_atari\log\Enduro\model_19959844_1348.pth'
	
	# agent.load(pre_model)
	# agent.train()
	agent.load_and_evaluate(pre_model)



