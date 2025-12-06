from .env_bataille import ParkingEnv
from .train_dqn import watch_trained_agent


watch_trained_agent("dqn_parking.pth", n_episodes=20)