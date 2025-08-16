# ===============================================================
# CartPole-v1 — DQN minimal (CPU friendly) avec explications
# ===============================================================

# ---- Imports standard ----
import os  # pour créer le dossier models/
import math  # pour des petites formules (ex: epsilon)
import random  # pour l'exploration et la reproductibilité
from collections import deque  # pour stocker les rewards récentes (log)

# ---- NumPy / PyTorch ----
import numpy as np  # tableaux et stats simples
import torch  # framework deep learning
import torch.nn as nn  # couches de réseau
import torch.optim as optim  # optimiseur (Adam)

# ---- Gymnasium (l'environnement CartPole) ----
import gymnasium as gym  # fournit "CartPole-v1"

# ------------- Hyperparamètres (tu peux jouer avec) -------------
SEED = 0                 # graine pour rendre (un peu) reproductible
TOTAL_STEPS = 50_000     # nombre total de steps d'entraînement
BUFFER_SIZE = 50_000     # taille max du replay buffer
BATCH_SIZE = 64          # taille des mini-batches d'apprentissage
GAMMA = 0.99             # facteur d'actualisation (valeur futur)
LR = 1e-3                # learning rate de l'optimiseur
LEARNING_STARTS = 1_000  # ne pas apprendre avant d'avoir assez d'expérience
TRAIN_FREQ = 1           # on fait une update tous les TRAIN_FREQ steps
TARGET_UPDATE_FREQ = 1_000  # copie du réseau vers la target (stabilité)
EVAL_FREQ = 5_000        # tous les EVAL_FREQ steps, on évalue sans exploration
EVAL_EPISODES = 5        # nb d'épisodes pour l'évaluation
# --- epsilon-greedy (exploration) ---
EPS_START = 1.0          # epsilon au début (100% random)
EPS_END = 0.05           # epsilon minimum (5% random)
EXPLORATION_FRACTION = 0.2  # on décroit epsilon sur 20% des steps (ici 10k steps)
# ---------------------------------------------------------------

# ---- Réseau Q (petit MLP) ----
class QNetwork(nn.Module):
    # Réseau 4 -> 128 -> 128 -> n_actions
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()  # init de nn.Module
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),  # couche linéaire: 4 -> 128
            nn.ReLU(),                # activation non-linéaire
            nn.Linear(128, 128),      # 128 -> 128
            nn.ReLU(),                # activation
            nn.Linear(128, n_actions) # 128 -> nb d'actions (2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, obs_dim] ; sortie: [batch, n_actions]
        return self.net(x)

# ---- Replay Buffer (mémoire d'expériences) ----
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        # On pré-alloue des tableaux NumPy pour l'efficacité
        self.capacity = capacity
        self.pos = 0  # index de la prochaine insertion (cercle)
        self.full = False  # devient True quand on a rempli au moins une fois

        # États, actions, rewards, next_states, done
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)  # 1.0 si fini, 0.0 sinon

    def add(self, state, action, reward, next_state, done):
        # On insère à la position courante (écrasement circulaire)
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = 1.0 if done else 0.0

        # On avance l'index circulaire
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def __len__(self):
        # Taille effective (avant d'être plein, c'est pos ; sinon capacity)
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int):
        # On échantillonne des indices aléatoires sans remise
        max_index = len(self)
        idxs = np.random.randint(0, max_index, size=batch_size)

        # On retourne des arrays prêts à passer en tenseurs
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

# ---- Scheduler epsilon (exploration -> exploitation) ----
def epsilon_by_step(step: int) -> float:
    # On décroît epsilon linéairement pendant EXPLORATION_FRACTION * TOTAL_STEPS
    decay_steps = int(EXPLORATION_FRACTION * TOTAL_STEPS)
    if step >= decay_steps:
        return EPS_END  # on a fini de décroitre
    # interpolation linéaire entre EPS_START et EPS_END
    frac = step / max(1, decay_steps)
    return EPS_START + frac * (EPS_END - EPS_START)

# ---- Politique: choisir une action (epsilon-greedy) ----
@torch.no_grad()  # pas de gradient pour l'acteur (choix d'action)
def select_action(qnet: QNetwork, state: np.ndarray, epsilon: float, n_actions: int) -> int:
    # Tirage aléatoire pour décider: explore ou exploite ?
    if random.random() < epsilon:
        return random.randrange(n_actions)  # action aléatoire (exploration)
    # Sinon: on exploite le réseau (argmax des Q-values)
    state_t = torch.from_numpy(state).float().unsqueeze(0)  # [1, obs_dim]
    q_values = qnet(state_t)  # [1, n_actions]
    action = int(torch.argmax(q_values, dim=1).item())  # action avec Q max
    return action

# ---- Évaluation (joue sans exploration) ----
@torch.no_grad()
def evaluate(qnet: QNetwork, episodes: int = EVAL_EPISODES) -> float:
    env = gym.make("CartPole-v1", render_mode=None)  # pas de fenêtre pour aller vite
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=SEED + 100 + ep)  # seed différente
        total = 0.0
        for _ in range(500):  # limite par sécurité
            a = select_action(qnet, obs, epsilon=0.0, n_actions=env.action_space.n)  # 0.0 => greedy pur
            obs, r, terminated, truncated, _ = env.step(a)
            total += float(r)
            if terminated or truncated:
                break
        scores.append(total)
    env.close()
    return float(np.mean(scores))

# ---- Boucle principale d'entraînement ----
def main():
    # 1) Seeds (un peu de reproductibilité)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 2) Env d'entraînement (sans fenêtre pour la vitesse)
    env = gym.make("CartPole-v1", render_mode=None)
    obs, _ = env.reset(seed=SEED)  # première obs
    obs_dim = env.observation_space.shape[0]  # 4
    n_actions = env.action_space.n  # 2

    # 3) Réseaux (online + target) + optimiseur + loss
    qnet = QNetwork(obs_dim, n_actions)            # réseau principal (online)
    target_qnet = QNetwork(obs_dim, n_actions)     # réseau cible (gelé entre updates)
    target_qnet.load_state_dict(qnet.state_dict()) # au début: identiques
    optimizer = optim.Adam(qnet.parameters(), lr=LR)  # Adam standard
    loss_fn = nn.SmoothL1Loss()                    # Huber loss (robuste)

    # 4) Replay buffer (mémoire d'expériences)
    buffer = ReplayBuffer(BUFFER_SIZE, obs_dim)

    # 5) Logs
    os.makedirs("models", exist_ok=True)          # dossier pour sauvegarder les modèles
    recent_rewards = deque(maxlen=20)             # moyenne glissante des 20 derniers épisodes
    best_eval = -float("inf")                     # meilleur score d'éval vu

    # 6) Variables d'épisode en cours
    episode_reward = 0.0                          # cumul de reward de l'épisode courant

    # 7) Boucle sur les steps globaux
    for step in range(1, TOTAL_STEPS + 1):
        # a) epsilon selon le step (exploration au début, puis exploitation)
        epsilon = epsilon_by_step(step)

        # b) choisir action (epsilon-greedy) depuis l'état courant
        action = select_action(qnet, obs, epsilon, n_actions)

        # c) faire un pas dans l'env
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)  # fin d'épisode ?

        # d) ajouter la transition au buffer
        buffer.add(obs, action, float(reward), next_obs, done)

        # e) avancer l'état, cumuler la reward
        obs = next_obs
        episode_reward += float(reward)

        # f) si épisode fini: reset + log
        if done:
            recent_rewards.append(episode_reward)      # stocke le score de l'épisode fini
            # petit log toutes les ~10 fins d'épisodes
            if len(recent_rewards) >= 5:
                print(f"[STEP {step:6d}] eps={epsilon:.2f} | mean_reward(20)={np.mean(recent_rewards):.1f}")
            # reset de l'épisode
            obs, _ = env.reset()
            episode_reward = 0.0

        # g) Apprentissage (après warmup + selon TRAIN_FREQ)
        if step >= LEARNING_STARTS and step % TRAIN_FREQ == 0 and len(buffer) >= BATCH_SIZE:
            # --- échantillonne un mini-batch ---
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            # --- passe en tenseurs PyTorch ---
            states_t = torch.from_numpy(states)               # [B, 4], float32
            actions_t = torch.from_numpy(actions).long()      # [B], int64
            rewards_t = torch.from_numpy(rewards)             # [B], float32
            next_states_t = torch.from_numpy(next_states)     # [B, 4], float32
            dones_t = torch.from_numpy(dones)                 # [B], float32 (0.0 ou 1.0)

            # --- Q(s,a) du réseau online ---
            q_values = qnet(states_t)                         # [B, n_actions]
            q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # sélectionne la colonne de 'actions'

            # --- max_a' Q_target(s', a') du réseau cible ---
            with torch.no_grad():
                next_q_values = target_qnet(next_states_t)    # [B, n_actions]
                next_q_max = next_q_values.max(dim=1).values  # [B]

                # --- Cible de Bellman : r + gamma * (1 - done) * max Q_target(s', a') ---
                target = rewards_t + GAMMA * (1.0 - dones_t) * next_q_max

            # --- Perte Huber et descente de gradient ---
            loss = loss_fn(q_sa, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)  # clip (sécurité)
            optimizer.step()

        # h) Mise à jour du réseau cible (hard update périodique)
        if step % TARGET_UPDATE_FREQ == 0:
            target_qnet.load_state_dict(qnet.state_dict())

        # i) Évaluation périodique (sans exploration)
        if step % EVAL_FREQ == 0:
            eval_mean = evaluate(qnet, episodes=EVAL_EPISODES)
            print(f"[EVAL @ {step:6d}] mean_reward={eval_mean:.1f} (best={best_eval:.1f})")
            # sauvegarde du meilleur
            if eval_mean > best_eval:
                best_eval = eval_mean
                torch.save(qnet.state_dict(), os.path.join("models", "best.pt"))
                print("  -> New best model saved at models/best.pt")

    # 8) Fin: sauvegarde du dernier modèle (pour reprendre)
    torch.save(qnet.state_dict(), os.path.join("models", "last.pt"))
    print("Training done. Model saved at models/last.pt")

# ---- Lancement du script ----
if __name__ == "__main__":
    main()
