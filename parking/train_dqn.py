import math
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt  # pour les graphiques

from .env_bataille import ParkingEnv
from .dqn_agent import DQN, ReplayBuffer, create_optimizer


def train_dqn(
    num_episodes=1000,
    gamma=0.99,
    batch_size=64,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.10,
    epsilon_decay=100_000,
    target_update_freq=20,
    buffer_capacity=100_000,
    convergence_window=100,      # fenêtre de moyenne glissante (épisodes)
    convergence_tol=1e-2,        # tolérance *relative* sur la variation (ex : 1e-2 = 1 %)
    convergence_patience=1000,   # nb de fois de suite avant arrêt
):
    """
    Entraîne un agent DQN sur ParkingEnv (places en bataille, 3 actions).
    Actions :
      0 : avancer + tourner à gauche
      1 : avancer tout droit
      2 : avancer + tourner à droite

    L'entraînement s'arrête quand le modèle est considéré comme convergé :
    - on suit la moyenne glissante des rewards sur `convergence_window` épisodes
    - si la variation relative de cette moyenne est < `convergence_tol`
      pendant `convergence_patience` mises à jour consécutives, on stoppe.
    """
    env = ParkingEnv()
    state = env.reset()
    state_dim = len(state)       # s'adapte à ton état (avec toutes les distances)
    action_dim = 3               # 0: gauche+avance, 1: tout droit, 2: droite+avance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Entraînement sur :", device)
    print("Dimension de l'état :", state_dim)
    print("Nombre d'actions :", action_dim)

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = create_optimizer(policy_net, lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    steps_done = 0
    episode_rewards = []

    # ---- Historique pour les graphiques ----
    epsilon_history = []   # epsilon au début de chaque épisode
    loss_history = []      # loss à chaque mise à jour

    # ---- Variables pour détecter la convergence ----
    prev_moving_avg = None
    stable_count = 0
    stopped_early = False


    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        epsilon_episode = None  # pour garder l'epsilon du début d'épisode

        while not done:
            # ----------------- Epsilon-greedy -----------------
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(
                -steps_done / epsilon_decay
            )
            steps_done += 1

            # on mémorise l'epsilon du début d'épisode pour le graphique
            if epsilon_episode is None:
                epsilon_episode = epsilon

            if random.random() < epsilon:
                # action aléatoire parmi {0,1,2}
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(s)
                    action = int(q_values.argmax(dim=1).item())

            # ----------------- Transition -----------------
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            buffer.push(state, action, reward, next_state, done)
            state = next_state

            # ----------------- Apprentissage -----------------
            if len(buffer) >= batch_size:
                (
                    states_b,
                    actions_b,
                    rewards_b,
                    next_states_b,
                    dones_b,
                ) = buffer.sample(batch_size)

                states_b = torch.tensor(states_b, device=device, dtype=torch.float32)
                actions_b = torch.tensor(actions_b, device=device).unsqueeze(1)
                rewards_b = torch.tensor(rewards_b, device=device, dtype=torch.float32).unsqueeze(1)
                next_states_b = torch.tensor(next_states_b, device=device, dtype=torch.float32)
                dones_b = torch.tensor(dones_b, device=device, dtype=torch.float32).unsqueeze(1)

                # Q(s,a) courant
                q_values = policy_net(states_b).gather(1, actions_b)

                # Q cible = r + gamma * max_a' Q_target(s', a') * (1 - done)
                with torch.no_grad():
                    max_next_q = target_net(next_states_b).max(dim=1, keepdim=True)[0]
                    target_q = rewards_b + gamma * max_next_q * (1.0 - dones_b)

                loss = nn.MSELoss()(q_values, target_q)
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        episode_rewards.append(total_reward)
        if epsilon_episode is not None:
            epsilon_history.append(epsilon_episode)

        # ----------------- Update du target_net -----------------
        if (episode + 1) % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # ----------------- Logs -----------------
        if (episode + 1) % 10 == 0:
            mean_last_50 = (
                np.mean(episode_rewards[-50:])
                if len(episode_rewards) >= 50
                else np.mean(episode_rewards)
            )
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Reward: {total_reward:.1f} | "
                f"moyenne(50 derniers): {mean_last_50:.3f} | "
                f"epsilon={epsilon:.3f}"
            )

        # ----------------- Détection de convergence -----------------
        if len(episode_rewards) >= convergence_window:
            current_moving_avg = np.mean(episode_rewards[-convergence_window:])

            if prev_moving_avg is not None:
                # variation *relative* (ex : 1e-2 = 1 %)
                diff_rel = abs(current_moving_avg - prev_moving_avg) / max(
                    1.0, abs(prev_moving_avg)
                )
                if diff_rel < convergence_tol:
                    stable_count += 1
                else:
                    stable_count = 0
            prev_moving_avg = current_moving_avg

            if stable_count >= convergence_patience:
                print(
                    f"Convergence détectée à l'épisode {episode + 1} : "
                    f"moyenne glissante ≈ {current_moving_avg:.3f}"
                )
                stopped_early = True
                break

    actual_episodes = len(episode_rewards)
    print("Entraînement terminé.")
    if stopped_early:
        print(f"Arrêt anticipé après {actual_episodes} épisodes (convergence).")
    else:
        print(f"Nombre total d'épisodes effectués : {actual_episodes}")

    # ----------------- Graphique 1 : moyenne glissante (convergence) -----------------
    if actual_episodes >= convergence_window:
        window = convergence_window
        moving_avgs = np.convolve(
            episode_rewards, np.ones(window) / window, mode="valid"
        )
        episodes_ma = np.arange(window, actual_episodes + 1)

        plt.figure()
        plt.plot(episodes_ma, moving_avgs)
        plt.xlabel("Épisode")
        plt.ylabel(f"Reward moyenne (fenêtre = {window})")
        plt.title("Convergence : moyenne glissante des rewards")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ----------------- Graphique 2 : epsilon par épisode -----------------
    if len(epsilon_history) > 0:
        plt.figure()
        episodes_eps = np.arange(1, len(epsilon_history) + 1)
        plt.plot(episodes_eps, epsilon_history)
        plt.xlabel("Épisode")
        plt.ylabel("ε (epsilon-greedy)")
        plt.title("Évolution d'epsilon au cours de l'entraînement")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ----------------- Graphique 3 : évolution de la loss -----------------
    if len(loss_history) > 0:
        window_loss = 100  # moyenne glissante pour lisser la loss
        if len(loss_history) >= window_loss:
            loss_avg = np.convolve(
                loss_history,
                np.ones(window_loss) / window_loss,
                mode="valid",
            )
            steps_loss = np.arange(window_loss, len(loss_history) + 1)
            eff_window = window_loss
        else:
            loss_avg = np.array(loss_history)
            steps_loss = np.arange(1, len(loss_history) + 1)
            eff_window = len(loss_history)

        plt.figure()
        plt.plot(steps_loss, loss_avg)
        plt.xlabel("Mise à jour du réseau (step)")
        plt.ylabel(f"Loss moyenne (fenêtre = {eff_window})")
        plt.title("Évolution de la loss du DQN")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Sauvegarde du modèle
    torch.save(policy_net.state_dict(), "dqn_parking.pth")
    print("Modèle sauvegardé dans dqn_parking.pth")

    return policy_net, episode_rewards


def watch_trained_agent(model_path="dqn_parking.pth", n_episodes=3):
    """
    Regarde l'agent entraîné jouer, avec rendu Pygame.
    """
    import pygame

    env = ParkingEnv()
    env._init_pygame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_state = env.reset()
    state_dim = len(dummy_state)
    action_dim = 3   # même chose : 0,1,2 uniquement

    policy_net = DQN(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        print(f"Episode de démonstration {ep + 1}/{n_episodes}")

        while not done:
            env.render()
            pygame.time.delay(30)

            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = policy_net(s)
                action = int(q_values.argmax(dim=1).item())

            state, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"Reward épisode démo: {total_reward:.1f}")

    pygame.quit()


if __name__ == "__main__":
    # 1) entraînement
    policy, rewards = train_dqn(num_episodes=30000)

    # 2) visualisation
    watch_trained_agent("dqn_parking.pth", n_episodes=3)