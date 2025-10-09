import numpy as np
import pygame
from gymnasium import spaces, Env

class GridWorldEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size=5, render_mode=None, cell_size=64, blocks=None, pit=None, pit_death_reward=-1.0):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode

        # Gym spaces
        self.action_space = spaces.Discrete(4)  # 0:Up,1:Down,2:Left,3:Right
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32
        )

        # Episode config
        self.goal_pos = [grid_size - 1, grid_size - 1]
        self.max_steps = grid_size * grid_size * 2
        self.steps = 0

        # Pygame handles (lazy-initialized)
        self._screen = None
        self._clock = None
        self._surface = None  # Offscreen surface for rgb_array
        self._font = None

        # Colors
        self._BG = (18, 18, 20)
        self._GRID = (60, 60, 70)
        self._AGENT = (66, 135, 245)
        self._GOAL = (90, 200, 90)
        self._BLOCK = (120, 120, 130)
        self._PIT = (200, 70, 70)      # lethal pit
        self._TXT = (230, 230, 235)

        # State
        self.agent_pos = [0, 0]
        self.state = None

        # Static blocks (impassable)
        if blocks is None:
            self.blocks = self._default_blocks()
        else:
            norm = {(int(r), int(c)) for (r, c) in blocks}
            self.blocks = {bc for bc in norm if bc not in {(0, 0), tuple(self.goal_pos)}}

        # Single lethal pit (fixed location each episode)
        self.pit_death_reward = float(pit_death_reward)
        if pit is None:
            self.pit = self._default_pit()
        else:
            pr, pc = int(pit[0]), int(pit[1])
            # ensure valid and not overlapping start/goal/blocks
            if (pr, pc) in self.blocks or (pr, pc) in {(0, 0), tuple(self.goal_pos)}:
                raise ValueError("Provided pit overlaps start/goal/blocks.")
            if not (0 <= pr < self.grid_size and 0 <= pc < self.grid_size):
                raise ValueError("Provided pit is out of bounds.")
            self.pit = (pr, pc)

    # --------------- Gym API ---------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        self.steps = 0
        self._rebuild_state()

        if self.render_mode == "human":
            self._init_pygame(human=True)
            self._draw()  # draw initial frame
            pygame.display.flip()

        return self.state.flatten(), {}

    def step(self, action):
        # Propose move within bounds first
        pr, pc = self.agent_pos
        if action == 0 and pr > 0:                      # Up
            pr -= 1
        elif action == 1 and pr < self.grid_size - 1:   # Down
            pr += 1
        elif action == 2 and pc > 0:                    # Left
            pc -= 1
        elif action == 3 and pc < self.grid_size - 1:   # Right
            pc += 1

        # Reject move if target is a block; otherwise commit
        if (pr, pc) not in self.blocks:
            self.agent_pos = [pr, pc]

        # Termination/truncation/reward
        self.steps += 1

        # Check lethal pit first
        on_pit = (self.agent_pos[0], self.agent_pos[1]) == self.pit
        if on_pit:
            # Step onto pit: immediate death
            reward = self.pit_death_reward
            terminated = True
            truncated = False
            self._rebuild_state()  # show final position on pit
            if self.render_mode is not None:
                frame = self.render()
                if self.render_mode == "rgb_array":
                    return self.state, reward, terminated, truncated, {"pixels": frame}
            return self.state.flatten(), reward, terminated, truncated, {}

        # Otherwise normal progression
        terminated = (self.agent_pos[0] == self.goal_pos[0]) and (self.agent_pos[1] == self.goal_pos[1])
        truncated = self.steps >= self.max_steps
        reward = 1.0 if terminated else -0.01

        # Update state grid
        self._rebuild_state()

        if self.render_mode is not None:
            frame = self.render()
            if self.render_mode == "rgb_array":
                return self.state, reward, terminated, truncated, {"pixels": frame}

        return self.state.flatten(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return

        human = self.render_mode == "human"
        self._init_pygame(human=human)
        self._handle_pygame_events()  # allow window close
        self._draw()

        if human:
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        else:
            if self._surface is None:
                w = self.grid_size * self.cell_size
                h = self.grid_size * self.cell_size
                self._surface = pygame.Surface((w, h))
            self._surface.blit(self._screen, (0, 0))
            pixels = pygame.surfarray.array3d(self._surface)  # (W,H,3)
            return np.transpose(pixels, (1, 0, 2))  # (H,W,3)

    def close(self):
        if self._screen is not None:
            pygame.display.quit()
        if self._clock is not None:
            pygame.quit()
        self._screen = None
        self._clock = None
        self._surface = None
        self._font = None

    # --------------- Pygame helpers ---------------

    def _init_pygame(self, human=True):
        if self._screen is not None:
            return
        pygame.init()
        w = self.grid_size * self.cell_size
        h = self.grid_size * self.cell_size
        flags = 0
        if human:
            self._screen = pygame.display.set_mode((w, h), flags)
            pygame.display.set_caption("GridWorld (Gymnasium + pygame)")
        else:
            self._screen = pygame.Surface((w, h))
        self._clock = pygame.time.Clock()
        try:
            self._font = pygame.font.SysFont("consolas", 16)
        except Exception:
            self._font = None

    def _handle_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def _draw(self):
        cs = self.cell_size
        g = self.grid_size
        screen = self._screen
        screen.fill(self._BG)

        # Grid lines
        for i in range(g + 1):
            pygame.draw.line(screen, self._GRID, (0, i * cs), (g * cs, i * cs), 1)
            pygame.draw.line(screen, self._GRID, (i * cs, 0), (i * cs, g * cs), 1)

        # Blocks (impassable)
        for (r, c) in self.blocks:
            pygame.draw.rect(
                screen,
                self._BLOCK,
                pygame.Rect(c * cs + 2, r * cs + 2, cs - 4, cs - 4),
                border_radius=6,
            )

        # Pit (lethal)
        pr, pc = self.pit
        pygame.draw.rect(
            screen,
            self._PIT,
            pygame.Rect(pc * cs + 6, pr * cs + 6, cs - 12, cs - 12),
            border_radius=10,
        )

        # Goal
        gx, gy = self.goal_pos
        pygame.draw.rect(
            screen,
            self._GOAL,
            pygame.Rect(gy * cs + 2, gx * cs + 2, cs - 4, cs - 4),
            border_radius=8,
        )

        # Agent
        ax, ay = self.agent_pos
        center = (ay * cs + cs // 2, ax * cs + cs // 2)
        pygame.draw.circle(screen, self._AGENT, center, cs // 3)

        # HUD
        if self._font is not None:
            txt = self._font.render(f"Steps: {self.steps}", True, self._TXT)
            screen.blit(txt, (8, 8))

    # --------------- Internals ---------------

    def _rebuild_state(self):
        """Encode: agent=4.0, goal=3.0, pit=2.0, block=1.0, empty=0.0"""
        g = self.grid_size
        s = np.zeros((g, g), dtype=np.float32)
        for (r, c) in self.blocks:
            s[r, c] = 1.0
        s[self.pit[0], self.pit[1]] = 2.0
        s[self.goal_pos[0], self.goal_pos[1]] = 3.0
        s[self.agent_pos[0], self.agent_pos[1]] = 4.0
        self.state = s

    def _default_blocks(self):
        """Deterministic wall one column left of center with a gap."""
        g = self.grid_size
        wall_col = max(1, (g // 2) - 1)
        gap_row = g // 2
        blocks = set()
        for r in range(1, g - 1):
            if r == gap_row:
                continue
            blocks.add((r, wall_col))
        blocks.discard((0, 0))
        blocks.discard((self.goal_pos[0], self.goal_pos[1]))
        return blocks

    def _default_pit(self):
        """
        Deterministic pit placement:
        - Prefer center-ish.
        - Never overlap start/goal/blocks.
        """
        candidates = [
            (self.grid_size // 2, self.grid_size // 2),
            (self.grid_size // 2, max(0, (self.grid_size // 2) - 1)),
            (max(1, self.grid_size // 2 - 1), self.grid_size // 2),
            (self.grid_size // 2, min(self.grid_size - 2, (self.grid_size // 2) + 1)),
        ]
        for rc in candidates:
            if rc not in self.blocks and rc not in {(0, 0), tuple(self.goal_pos)}:
                return rc
        # Fallback: scan
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self.blocks and (r, c) not in {(0, 0), tuple(self.goal_pos)}:
                    return (r, c)
        raise RuntimeError("Could not place pit without overlap.")

# ---------------- Demo ----------------
if __name__ == "__main__":
    env = GridWorldEnv(grid_size=6, render_mode="human", cell_size=72)
    obs, info = env.reset()
    total_reward = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                raise SystemExit

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0
