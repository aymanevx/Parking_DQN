from .train_dqn import watch_trained_agent


watch_trained_agent("dqn_parking.pth", n_episodes=3,save_gif=True)