import math
import random
import numpy as np
import pygame


# ============================================================
# 1) ENVIRONNEMENT DE PARKING (PLACES EN BATAILLE)
# ============================================================

class ParkingEnv:
    """
    Version simplifiée de Dr Parking (vue de dessus), PLACES EN BATAILLE.

    - Parking rectangulaire 10 x 16 unités
    - Deux colonnes de places en bataille (à gauche et à droite de l'allée)
    - Presque toutes les places sont occupées par des voitures garées
    - Une place vide = objectif à atteindre
    - Ta voiture part dans l'allée centrale, orientée vers le haut
    - À chaque partie, la place vide change d'endroit
    """

    def __init__(self):
        # Taille logique du parking
        self.width = 10.0   # axe X
        self.height = 16.0  # axe Y

        # Position de départ de ta voiture (dans l'allée centrale, orientée vers le haut)
        self.start_x = self.width / 2.0  # milieu en X
        self.start_y = 2.0               # proche du bas
        self.start_angle = math.pi / 2.0  # vers le haut

        # Dimensions de ta voiture (en unités monde)
        self.car_length = 1.3
        self.car_width = 0.7

        # Dynamique
        self.forward_step = 0.5
        self.turn_angle = math.radians(10)

        # Slots de parking (colonnes gauche/droite)
        # Chaque slot : (x_min, y_min, x_max, y_max, is_goal)
        self.slots = []
        self._build_slots()

        # Attributs liés à la place cible
        self.goal_slot = None
        self.goal_center = None
        # tolérance sur l’angle pour être “horizontalement garée”
        self.goal_angle_tolerance = math.radians(18)

        # Obstacles = voitures garées (remplis après choix du slot)
        self.obstacles = []

        # Durée max d'un épisode
        self.max_steps = 100

        # Statut (pour affichage : "playing" / "win" / "collision")
        self.status = "playing"

        # Pour savoir si la voiture était dans la place au step précédent
        self.in_slot_last_step = False

        # Pygame
        self._pygame_initialized = False

        # Premier reset (choisit aussi une place vide aléatoire)
        self.reset()

    # --------------------------------------------------------
    # Construction des places de parking EN BATAILLE
    # --------------------------------------------------------

    def _build_slots(self):
        """
        Construit 2 colonnes de 3 places EN BATAILLE :
        - colonne gauche : x in [1, 3]
        - colonne droite : x in [7, 9]
        - 3 places empilées verticalement de chaque côté de l'allée
        """
        self.slots = []

        slot_width = 2.0       # largeur sur l'axe X
        slot_height = 2.5      # hauteur sur l'axe Y
        y_start = 5.0          # début en Y
        gap = 0.0              # espace entre places

        # Colonne de gauche
        x_min_left = 1.0
        x_max_left = x_min_left + slot_width

        # Colonne de droite
        x_max_right = self.width - 1.0
        x_min_right = x_max_right - slot_width

        for i in range(3):
            y_min = y_start + i * (slot_height + gap)
            y_max = y_min + slot_height

            # place gauche
            self.slots.append((x_min_left,  y_min, x_max_left,  y_max, False))
            # place droite
            self.slots.append((x_min_right, y_min, x_max_right, y_max, False))

    def _choose_random_goal_slot(self):
        """
        Choisit UNE place comme slot vert (vide),
        met à jour self.slots, self.goal_slot, self.goal_center
        et reconstruit les obstacles (voitures garées).
        """
        idx = random.randrange(len(self.slots))   # aléatoire
        # Place fixe pour l'entraînement :
        # idx = 4

        new_slots = []
        for i, (x_min, y_min, x_max, y_max, _) in enumerate(self.slots):
            is_goal = (i == idx)
            new_slots.append((x_min, y_min, x_max, y_max, is_goal))
        self.slots = new_slots

        gx_min, gy_min, gx_max, gy_max, _ = self.slots[idx]
        self.goal_slot = (gx_min, gy_min, gx_max, gy_max)
        self.goal_center = np.array(
            [(gx_min + gx_max) / 2.0, (gy_min + gy_max) / 2.0],
            dtype=np.float32
        )

        # Reconstruire les voitures garées (toutes les places sauf la goal)
        self.obstacles = self._build_obstacles_from_slots()

    def _build_obstacles_from_slots(self):
        """
        Crée des voitures garées à l'intérieur des slots NON cibles.
        Chaque obstacle est légèrement plus petit que le slot.
        """
        obstacles = []
        margin_x = 0.2
        margin_y = 0.25

        for (x_min, y_min, x_max, y_max, is_goal) in self.slots:
            if is_goal:
                continue  # on laisse ce slot vide pour la place de parking cible
            car_x_min = x_min + margin_x
            car_x_max = x_max - margin_x
            car_y_min = y_min + margin_y
            car_y_max = y_max - margin_y
            obstacles.append(np.array([car_x_min, car_y_min, car_x_max, car_y_max]))
        return obstacles

    # --------------------------------------------------------
    # Utilitaires géométriques : coins de la voiture
    # --------------------------------------------------------

    def _car_corners(self, x, y, angle):
        """
        Renvoie les coordonnées (x,y) des 4 coins de la voiture
        en unités du monde (PAS en pixels).
        """
        L = self.car_length
        Wc = self.car_width

        local_corners = [
            (-L / 2, -Wc / 2),
            ( L / 2, -Wc / 2),
            ( L / 2,  Wc / 2),
            (-L / 2,  Wc / 2),
        ]

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        world_corners = []
        for (lx, ly) in local_corners:
            wx = x + lx * cos_a - ly * sin_a
            wy = y + lx * sin_a + ly * cos_a
            world_corners.append((wx, wy))
        return world_corners

    def _obstacle_vectors(self, x, y):
        """
        Retourne une liste de (dx, dy, dist) pour TOUS les obstacles.
        dx = obstacle_centre.x - voiture.x
        dy = obstacle_centre.y - voiture.y
        dist = sqrt(dx^2 + dy^2)
        L'ordre est simplement l'ordre des obstacles dans self.obstacles.
        """
        vectors = []
        for (x_min, y_min, x_max, y_max) in self.obstacles:
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            dx = cx - x
            dy = cy - y
            dist = math.hypot(dx, dy)
            vectors.append((dx, dy, dist))
        return vectors

    # --------------------------------------------------------
    # Logique (état, reset, step)
    # --------------------------------------------------------

    def reset(self):
        # À chaque partie : (re)choisit une place vide
        self._choose_random_goal_slot()

        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.steps = 0
        self.status = "playing"
        self.in_slot_last_step = False
        return self._get_state()

    def _get_state(self):
        """
        État renvoyé :
        [
            x, y, angle,
            dx_goal, dy_goal, dist_goal,
            dx_obs_0, dy_obs_0, dist_obs_0,
            dx_obs_1, dy_obs_1, dist_obs_1,
            ...
            dx_obs_n, dy_obs_n, dist_obs_n
        ]
        """
        x = self.x
        y = self.y
        angle = self.angle

        dx_goal = float(self.goal_center[0] - x)
        dy_goal = float(self.goal_center[1] - y)
        dist_goal = math.hypot(dx_goal, dy_goal)

        state_list = [
            float(x),
            float(y),
            float(angle),
            dx_goal,
            dy_goal,
            float(dist_goal),
        ]

        obs_vectors = self._obstacle_vectors(x, y)
        for (dx_obs, dy_obs, dist_obs) in obs_vectors:
            state_list.append(float(dx_obs))
            state_list.append(float(dy_obs))
            state_list.append(float(dist_obs))

        return np.array(state_list, dtype=np.float32)

    def _check_collision(self, x, y, angle):
        """
        Collision si AU MOINS UN COIN de la voiture :
        - sort du parking
        - rentre dans une voiture garée
        """
        corners = self._car_corners(x, y, angle)

        # Bord du parking
        for (cx, cy) in corners:
            if cx < 0 or cx > self.width or cy < 0 or cy > self.height:
                return True

        # Collision avec voitures garées
        for obs in self.obstacles:
            x_min, y_min, x_max, y_max = obs
            for (cx, cy) in corners:
                if x_min <= cx <= x_max and y_min <= cy <= y_max:
                    return True

        return False

    # ---------------- Orientation & parking ------------------

    def _is_orientation_horizontal(self, angle):
        """
        Vrai si la voiture est plus ou moins horizontale :
        angle proche de 0 ou de pi.
        """
        candidates = [0.0, math.pi]
        min_diff = float("inf")
        for target in candidates:
            diff = abs((angle - target + math.pi) % (2 * math.pi) - math.pi)
            min_diff = min(min_diff, diff)
        return min_diff < self.goal_angle_tolerance

    def _is_in_slot(self, x, y, angle):
        """
        Version simplifiée : juste le centre de la voiture dans la place.
        Sert de palier intermédiaire de reward.
        """
        gx_min, gy_min, gx_max, gy_max = self.goal_slot
        return gx_min <= x <= gx_max and gy_min <= y <= gy_max

    def _is_parked(self, x, y, angle):
        """
        Garée = TOUS les coins de la voiture sont à l'intérieur
        du slot vert (goal_slot) ET orientation horizontale.
        """
        gx_min, gy_min, gx_max, gy_max = self.goal_slot
        corners = self._car_corners(x, y, angle)

        for (cx, cy) in corners:
            if not (gx_min <= cx <= gx_max and gy_min <= cy <= gy_max):
                return False

        # Vérifier l'orientation (horizontale)
        if not self._is_orientation_horizontal(angle):
            return False

        return True

    def step(self, action: int):
        """
        Actions :
            0 : avancer + tourner à gauche
            1 : avancer tout droit
            2 : avancer + tourner à droite
        """
        if self.status != "playing":
            return self._get_state(), 0.0, True, {}

        self.steps += 1
        x, y, angle = self.x, self.y, self.angle

        # Est-ce qu'on était dans la place au step précédent ?
        prev_in_slot = self.in_slot_last_step

        # Distance à la place AVANT le mouvement
        old_dx = float(self.goal_center[0] - x)
        old_dy = float(self.goal_center[1] - y)
        old_dist = math.hypot(old_dx, old_dy)

        # -----------------------------
        # Dynamique de mouvement
        # -----------------------------
        if action == 0:        # gauche + avance
            angle += self.turn_angle
            x += self.forward_step * math.cos(angle)
            y += self.forward_step * math.sin(angle)
        elif action == 1:      # tout droit
            x += self.forward_step * math.cos(angle)
            y += self.forward_step * math.sin(angle)
        elif action == 2:      # droite + avance
            angle -= self.turn_angle
            x += self.forward_step * math.cos(angle)
            y += self.forward_step * math.sin(angle)
        else:
            raise ValueError("Action invalide (0,1,2 uniquement)")

        # Normalisation de l'angle
        angle = (angle + math.pi) % (2 * math.pi) - math.pi

        # Distance APRÈS le mouvement
        new_dx = float(self.goal_center[0] - x)
        new_dy = float(self.goal_center[1] - y)
        new_dist = math.hypot(new_dx, new_dy)

        # Est-ce qu'on est dans la place APRÈS le mouvement ?
        in_slot_now = self._is_in_slot(x, y, angle)

        # -----------------------------
        # Récompenses + statut RL
        # -----------------------------
        reward = 0.0

        # Collision ?
        if self._check_collision(x, y, angle):
            reward = -1.0
            done = True
            self.status = "collision"

        # Garée ?
        elif self._is_parked(x, y, angle):
            reward = +100000.0
            done = True
            self.status = "win"

        # Était dans la place et vient d'en sortir → énorme punition
        elif prev_in_slot and not in_slot_now:
            reward = -300.0
            done = True
            self.status = "collision"

        # Dans la place, mais pas encore “parking parfait”
        elif in_slot_now:
            reward += 0.7
            done = False
            self.status = "playing"

        else:
            # Shaping de distance : se rapprocher = reward positive
            delta = old_dist - new_dist  # >0 si on se rapproche
            reward += 0.2 * delta
            reward -= 0.001  # pénalité de temps

            done = (self.steps >= self.max_steps)
            if done:
                reward = -100.0
                self.status = "collision"
            else:
                self.status = "playing"

        # Mise à jour
        self.x, self.y, self.angle = x, y, angle
        self.in_slot_last_step = in_slot_now

        return self._get_state(), reward, done, {}

    # --------------------------------------------------------
    # Pygame : init + rendu
    # --------------------------------------------------------

    def _init_pygame(self):
        pygame.init()
        self.screen_size = 720
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Dr Parking - Places en bataille")
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("arial", 72, bold=True)
        self.font_small = pygame.font.SysFont("arial", 32, bold=True)
        self._pygame_initialized = True

    def render(self):
        if not self._pygame_initialized:
            self._init_pygame()

        screen = self.screen
        W = self.screen_size
        H = self.screen_size
        scale_x = W / self.width
        scale_y = H / self.height

        ASPHALT = (30, 30, 30)
        BORDER = (250, 210, 70)
        LANE_LINE = (240, 240, 240)
        SLOT_LINE = (220, 220, 220)
        GOAL_LINE = (50, 220, 50)
        PARKED_COLORS = [
            (180, 60, 60),
            (60, 140, 210),
            (220, 180, 60),
            (150, 80, 190),
        ]
        CAR = (220, 50, 50)

        screen.fill(ASPHALT)
        pygame.draw.rect(screen, BORDER, (0, 0, W, H), width=6)

        lane_y_min = 0.0 * scale_y
        lane_y_max = self.height * scale_y
        lane_x_mid = (self.width / 2.0) * scale_x
        dash_len = 30
        gap = 20
        y = 0
        while y < lane_y_max:
            pygame.draw.line(
                screen,
                LANE_LINE,
                (lane_x_mid, y),
                (lane_x_mid, min(y + dash_len, lane_y_max)),
                width=3,
            )
            y += dash_len + gap

        # Places
        for (x_min, y_min, x_max, y_max, is_goal) in self.slots:
            rx = int(x_min * scale_x)
            ry = int(y_min * scale_y)
            rw = int((x_max - x_min) * scale_x)
            rh = int((y_max - y_min) * scale_y)
            color = GOAL_LINE if is_goal else SLOT_LINE
            width = 4 if is_goal else 2
            pygame.draw.rect(screen, color, (rx, ry, rw, rh), width=width)

        # Voitures garées
        for i, obs in enumerate(self.obstacles):
            x_min, y_min, x_max, y_max = obs
            rect = pygame.Rect(
                int(x_min * scale_x),
                int(y_min * scale_y),
                int((x_max - x_min) * scale_x),
                int((y_max - y_min) * scale_y),
            )
            col = PARKED_COLORS[i % len(PARKED_COLORS)]
            pygame.draw.rect(screen, col, rect)
            pygame.draw.rect(screen, SLOT_LINE, rect, width=2)

        # Ta voiture
        cx = self.x * scale_x
        cy = self.y * scale_y
        L = self.car_length * scale_x
        Wc = self.car_width * scale_y
        angle = self.angle

        local_corners = [
            (-L / 2, -Wc / 2),
            ( L / 2, -Wc / 2),
            ( L / 2,  Wc / 2),
            (-L / 2,  Wc / 2),
        ]

        car_points = []
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        for (lx, ly) in local_corners:
            x_rot = cx + lx * cos_a - ly * sin_a
            y_rot = cy + lx * sin_a + ly * cos_a
            car_points.append((int(x_rot), int(y_rot)))

        pygame.draw.polygon(screen, CAR, car_points)
        pygame.draw.polygon(screen, (255, 255, 255), car_points, width=2)

        nose_len = L * 0.6
        nose_x = cx + nose_len * math.cos(angle)
        nose_y = cy + nose_len * math.sin(angle)
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (int(cx), int(cy)),
            (int(nose_x), int(nose_y)),
            width=2,
        )

        if self.status in ("win", "collision"):
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))

            box_w, box_h = 520, 220
            box_rect = pygame.Rect(0, 0, box_w, box_h)
            box_rect.center = (W // 2, H // 2)

            if self.status == "win":
                box_color = (0, 200, 150)
                title_text = "FÉLICITATIONS !"
                subtitle = "Vous êtes parfaitement garée."
            else:
                box_color = (220, 50, 70)
                title_text = "GAME OVER"
                subtitle = "Vous avez heurté une voiture ou le bord."

            pygame.draw.rect(screen, box_color, box_rect, border_radius=25)
            pygame.draw.rect(screen, (255, 255, 255), box_rect, width=4, border_radius=25)

            text_main = self.font_big.render(title_text, True, (20, 20, 20))
            text_sub = self.font_small.render(subtitle, True, (20, 20, 20))
            text_hint = self.font_small.render(
                "Appuyez sur R pour recommencer",
                True,
                (255, 255, 255),
            )

            rect_main = text_main.get_rect(center=(W // 2, H // 2 - 30))
            rect_sub = text_sub.get_rect(center=(W // 2, H // 2 + 35))
            rect_hint = text_hint.get_rect(center=(W // 2, H // 2 + 90))

            screen.blit(text_main, rect_main)
            screen.blit(text_sub, rect_sub)
            screen.blit(text_hint, rect_hint)

        pygame.display.flip()
        self.clock.tick(30)


# ============================================================
# 2) MODE JEU MANUEL UNIQUEMENT
# ============================================================

def manual_game():
    env = ParkingEnv()
    env._init_pygame()
    state = env.reset()
    done = False
    running = True

    print("Contrôles :")
    print("  ← : avancer + tourner à gauche")
    print("  ↑ : avancer tout droit")
    print("  → : avancer + tourner à droite")
    print("  R : reset")
    print("  ESC : quitter")

    while running:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = env.reset()
                    done = False
                elif not done:
                    if event.key == pygame.K_LEFT:
                        action = 0
                    elif event.key == pygame.K_UP:
                        action = 1
                    elif event.key == pygame.K_RIGHT:
                        action = 2

        if action is not None and not done:
            state, reward, done, _ = env.step(action)

        env.render()

    pygame.quit()


# ============================================================
# 3) MAIN
# ============================================================

if __name__ == "__main__":
    manual_game()
