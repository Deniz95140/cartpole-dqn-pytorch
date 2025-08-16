
# 🎮 CartPole DQN – PyTorch 🏋️‍♂️🧠

Un projet d’intelligence artificielle où une IA apprend à résoudre l’environnement **CartPole-v1** (la tige à équilibrer sur un chariot) en utilisant le **Deep Q-Learning (DQN)** avec **PyTorch**.

---

## 📂 Contenu du projet

- `train_dqn.py` → Script principal pour entraîner l’IA avec DQN
- `play.py` → Lancer une partie avec le modèle déjà entraîné
- `models/best.pt` → Modèle entraîné avec les meilleures performances
- `models/last.pt` → Dernier modèle sauvegardé (peut être moins bon)
- `requirements.txt` → Les librairies Python nécessaires au projet

---

## 🎯 Objectif

Apprendre à une IA à maintenir la barre en équilibre le plus longtemps possible, en déplaçant le chariot à gauche ou à droite.  
L'agent s'améliore à chaque épisode en apprenant de ses erreurs.

---

## 🛠️ Technologies utilisées

- Python 3.10+
- PyTorch (réseaux de neurones)
- Gymnasium (environnement CartPole-v1)
- NumPy

---

## ⚙️ Installation

```bash
# 1. Ouvrir un terminal dans le dossier du projet
# 2. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Entraîner l’IA depuis zéro

```bash
python train_dqn.py
```

### Faire jouer l’IA avec le meilleur modèle

```bash
python play.py
```

---

## 🧠 Fonctionnement

L’agent IA utilise l’algorithme **Deep Q-Learning** pour apprendre à contrôler le chariot et garder la barre en équilibre.  
À chaque action, il reçoit une récompense. Il apprend à maximiser cette récompense au fil du temps, jusqu’à résoudre l’environnement.

---

## ✍️ Auteur

Projet réalisé par **Deniz** aka **DebugosaurusRex** 🦖  
Portfolio & autres projets IA à retrouver sur [ton GitHub](https://github.com/TON_PSEUDO)

