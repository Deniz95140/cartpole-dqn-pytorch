import gymnasium as gym
import numpy as np

N_EPISODES = 10
MAX_STEPS = 500
RENDER = False  # mets False si tu veux aller vite sans fenêtre

def main():
    episode_returns = []

    # Définir proprement le mode de rendu (fenêtre ou non)
    render_mode = "human" if RENDER else None

    # Correct: CartPole-v1
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # Espaces observation/action (diagnostic)
    obs_space = env.observation_space   # 4 floats: pos, vit; angle, vit angulaire
    act_space = env.action_space        # 2 actions: gauche (0) ou droite (1)

    print("[INFO] Observation space:", obs_space)
    print("[INFO] Action space:", act_space)

    for ep in range(1, N_EPISODES + 1):
        # seed=ep pour un minimum de reproductibilité
        obs, info = env.reset(seed=ep)
        total_reward = 0.0

        for t in range(MAX_STEPS):
            # Agent aléatoire (baseline)
            action = act_space.sample()

            # Un pas d'environnement
            new_obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            obs = new_obs

            # Fin de l’épisode: bâton tombé (terminated) ou limite atteinte (truncated)
            if terminated or truncated:
                break

        episode_returns.append(total_reward)
        print(f"[RAND] Episode {ep:02d} - Reward: {total_reward:.1f} - Steps: {t+1}")

    # Stat simple sur N épisodes
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    print(f"[RAND] Moyenne sur {N_EPISODES} épisodes: {mean_return:.1f} ± {std_return:.1f}")

    env.close()

if __name__ == "__main__":
    main()
