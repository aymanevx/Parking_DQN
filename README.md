# Parking_DQN

Projet de **Deep Reinforcement Learning** utilisant un **Deep Q-Network (DQN)** pour apprendre à un agent à effectuer un stationnement automatique dans un environnement simulé.

##  Objectif
Entraîner un agent capable de se garer de manière autonome en apprenant les actions optimales (déplacements, orientation, etc.) à partir de récompenses, sans règles codées manuellement.

##  Démonstration
![Parking DQN Demo](assets/demo.gif)

##  Structure du projet
```text
Parking_DQN/
├── parking/        # Environnement + agent DQN
├── assets/         # Images / GIFs (démonstration)
│   └── demo.gif
├── .gitignore
└── README.md
```

##  Méthode
- Environnement de stationnement simulé  
- Agent entraîné avec l’algorithme **Deep Q-Learning (DQN)**  
- Stratégie d’exploration / exploitation (epsilon-greedy)  
- Fonction de récompense guidant l’agent vers un stationnement réussi 

##  Utilisation

### Cloner le projet
```bash
git clone https://github.com/aymanevx/Parking_DQN.git
cd Parking_DQN
```
### Lancer l’entraînement
```bash
python -m parking.train_dqn
```

### Visualiser l'agent entrainé
```bash
python -m parking.watch
```
### Paramètres principaux

Nombre d’épisodes

Taux d’apprentissage

Facteur de discount (gamma)

Epsilon (exploration)

Ces paramètres peuvent être ajustés pour améliorer la convergence et les performances de l’agent.

##  Remarques

Ce projet est à visée pédagogique et expérimentale, et peut être étendu avec :

Une reward fonction plus complexe

D’autres architectures de réseaux neuronaux

Des environnements de stationnement plus difficiles
