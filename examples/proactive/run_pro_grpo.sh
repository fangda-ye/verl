python -m verl.trainer.main_ppo \
    --config-path examples/proactive \
    --config-name config_with_group_aware_reward \
    # ++reward_model.reward_manager=group_aware \
    # ++custom_reward_function.path=/path/to/my_reward.py \
    # ++custom_reward_function.name=adaptive_reward
