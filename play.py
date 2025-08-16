# ===============================
# CartPole-v1 — PLAY (greedy)
# Charge le modèle et affiche le jeu
# ===============================

import argparse              # lire les arguments en ligne de commande
import time                  # petite pause pour voir l'animation
import os                    # vérifier si le fichier modèle existe

import gymnasium as gym      # l'environnement CartPole
import torch
import torch.nn as nn        # pour redéfinir le même réseau que dans train_dqn

# --- Réseau Q (même architecture que dans train_dqn.py) ---
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def select_action_greedy(qnet: QNetwork, state, n_actions: int) -> int:
    """Choisit l'action avec la plus grande Q-valeur (pas d'exploration)."""
    state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
    q_values = qnet(state_t)                                            # [1, n_actions]
    action = int(torch.argmax(q_values, dim=1).item())                  # indice max
    return action

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best.pt", help="Chemin du modèle .pt")
    parser.add_argument("--episodes", type=int, default=5, help="Nombre d'épisodes à jouer")
    parser.add_argument("--sleep", type=float, default=0.01, help="Pause (secondes) entre pas pour voir l'anim")
    parser.add_argument("--seed", type=int, default=123, help="Seed de départ (facultatif)")
    args = parser.parse_args()

    # Si best.pt n'existe pas, on tente last.pt
    model_path = args.model
    if not os.path.isfile(model_path):
        fallback = "models/last.pt"
        if os.path.isfile(fallback):
            print(f"[INFO] Modèle {model_path} introuvable. Utilisation de {fallback}.")
            model_path = fallback
        else:
            raise FileNotFoundError("Aucun modèle trouvé (ni best.pt ni last.pt). Entraîne d'abord train_dqn.py.")

    # Création de l'env AVEC FENÊTRE (render_mode='human')
    env = gym.make("CartPole-v1", render_mode="human")

    # Dimensions (CartPole: obs_dim=4, n_actions=2)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Chargement du réseau
    qnet = QNetwork(obs_dim, n_actions)
    qnet.load_state_dict(torch.load(model_path, map_location="cpu"))
    qnet.eval()  # mode évaluation (pas de dropout/batchnorm ici mais bonne pratique)

    # Boucle de jeu
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)  # reset; seed juste pour la forme
        total = 0.0
        steps = 0

        while True:
            # Action "greedy" (0% d'exploration)
            action = select_action_greedy(qnet, obs, n_actions)

            # Avancer d'un pas
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1

            # Petite pause pour voir l'animation (sinon c'est ultra rapide)
            time.sleep(args.sleep)

            # Fin d'épisode ?
            if terminated or truncated:
                print(f"[PLAY] Episode {ep} — Reward: {total:.1f} — Steps: {steps}")
                break

    env.close()

if __name__ == "__main__":
    main()
