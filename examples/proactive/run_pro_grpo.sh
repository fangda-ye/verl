python -m verl.trainer.main_ppo \
    --config-path your/config/path \
    --config-name your_config \
    ++reward_model.reward_manager=group_aware \
    ++custom_reward_function.path=/path/to/my_reward.py \
    ++custom_reward_function.name=adaptive_reward
