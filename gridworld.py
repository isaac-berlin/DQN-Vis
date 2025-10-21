import numpy as np
import pygame
import os
from gymnasium import spaces, Env

class GridWorldEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size=6, render_mode=None, cell_size=64, pit_death_reward=-1.0):
        super().__init__()
        if grid_size < 6:
            raise ValueError("grid_size must be >= 6 to keep the static layout used here.")
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode

        # Actions: 0:Up,1:Down,2:Left,3:Right
        self.action_space = spaces.Discrete(4)

        # Observation: (row, col) of the agent
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([grid_size - 1, grid_size - 1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Episode config
        self.goal_pos = [grid_size - 1, grid_size - 1]
        self.max_steps = grid_size * grid_size * 2
        self.steps = 0

        # Pygame handles
        self._screen = None
        self._clock = None
        self._surface = None

        # Colors
        self._BG = (18, 18, 20)
        self._GRID = (60, 60, 70)
        self._AGENT = (66, 135, 245)
        self._GOAL = (90, 200, 90)
        self._BLOCK = (120, 120, 130)
        self._PIT = (200, 70, 70)

        # State
        self.agent_pos = [0, 0]

        # --- Static layout (same as your current default for 6x6) ---
        # Impassable blocks form a partial vertical wall at column 2 with a gap at row 3
        self.blocks = {(1, 2), (2, 2), (4, 2)}
        # Lethal pit at center-ish
        self.pit_death_reward = pit_death_reward
        self.pit = (3, 3)

        # Sanity: ensure blocks/pit are inside bounds and not overlapping start/goal
        for (r, c) in self.blocks:
            if not (0 <= r < grid_size and 0 <= c < grid_size):
                raise ValueError("A static block is out of bounds for this grid_size.")
        if not (0 <= self.pit[0] < grid_size and 0 <= self.pit[1] < grid_size):
            raise ValueError("Static pit is out of bounds for this grid_size.")
        if (0, 0) in self.blocks or tuple(self.goal_pos) in self.blocks:
            raise ValueError("Static blocks overlap start/goal.")
        if self.pit in self.blocks or self.pit in {(0, 0), tuple(self.goal_pos)}:
            raise ValueError("Static pit overlaps start/goal/blocks.")

    # --------------- Gym API ---------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        self.steps = 0

        if self.render_mode == "human":
            self._init_pygame(human=True)
            self._draw()
            pygame.display.flip()

        obs = np.array(self.agent_pos, dtype=np.float32)
        return obs, {}

    def step(self, action):
        pr, pc = self.agent_pos

        if action == 0 and pr > 0:                      # Up
            pr -= 1
        elif action == 1 and pr < self.grid_size - 1:   # Down
            pr += 1
        elif action == 2 and pc > 0:                    # Left
            pc -= 1
        elif action == 3 and pc < self.grid_size - 1:   # Right
            pc += 1

        # Blocked?
        if (pr, pc) not in self.blocks:
            self.agent_pos = [pr, pc]

        self.steps += 1

        # Pit check
        on_pit = (self.agent_pos[0], self.agent_pos[1]) == self.pit
        if on_pit:
            reward = float(self.pit_death_reward)
            terminated = True
            truncated = False
            frame = self.render() if self.render_mode is not None else None
            obs = np.array(self.agent_pos, dtype=np.float32)
            if self.render_mode == "rgb_array":
                return obs, reward, terminated, truncated, {"pixels": frame}
            return obs, reward, terminated, truncated, {}

        # Goal / timeout
        terminated = (self.agent_pos[0] == self.goal_pos[0]) and (self.agent_pos[1] == self.goal_pos[1])
        truncated = self.steps >= self.max_steps
        reward = 1.0 if terminated else -0.01

        frame = self.render() if self.render_mode is not None else None
        obs = np.array(self.agent_pos, dtype=np.float32)
        if self.render_mode == "rgb_array":
            return obs, reward, terminated, truncated, {"pixels": frame}
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return

        human = self.render_mode == "human"
        self._init_pygame(human=human)

        # For human mode we should allow closing the window.
        if human:
            self._handle_pygame_events()

        # Draw the frame onto self._screen (which points to the right surface)
        self._draw()

        if human:
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return  # human mode returns None (standard Gym/Gymnasium pattern)

        # rgb_array mode: grab pixels from the offscreen surface
        surf = self._screen  # offscreen surface
        # shape (W, H, 3) -> transpose to (H, W, 3)
        pixels = pygame.surfarray.array3d(surf).transpose(1, 0, 2).copy()
        return pixels
    
    def close(self):
        # Clean up both human and rgb surfaces safely
        try:
            if self._screen is not None and isinstance(self._screen, pygame.Surface):
                # If we created a window, quit the display; if offscreen, this is harmless.
                pygame.display.quit()
        finally:
            pygame.quit()
            self._screen = None
            self._clock = None
            self._surface = None
            if hasattr(self, "_rgb_surface"):
                self._rgb_surface = None
    # --------------- Pygame helpers ---------------

    def _init_pygame(self, human=True):
        if self._screen is not None or self._surface is not None:
            return
        pygame.init()
        w = self.grid_size * self.cell_size
        h = self.grid_size * self.cell_size

        if human:
            # Onscreen window for human mode
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("GridWorld")
        else:
            # Offscreen surface for rgb_array mode
            self._rgb_surface = pygame.Surface((w, h))
            self._screen = self._rgb_surface  # draw onto this

        self._clock = pygame.time.Clock()

    def _handle_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def _draw(self):
        cs = self.cell_size
        g = self.grid_size
        screen = self._screen
        screen.fill(self._BG)

        # Grid
        for i in range(g + 1):
            pygame.draw.line(screen, self._GRID, (0, i * cs), (g * cs, i * cs), 1)
            pygame.draw.line(screen, self._GRID, (i * cs, 0), (i * cs, g * cs), 1)

        # Blocks
        for (r, c) in self.blocks:
            pygame.draw.rect(screen, self._BLOCK, pygame.Rect(c * cs + 2, r * cs + 2, cs - 4, cs - 4), border_radius=6)

        # Pit
        pr, pc = self.pit
        pygame.draw.rect(screen, self._PIT, pygame.Rect(pc * cs + 6, pr * cs + 6, cs - 12, cs - 12), border_radius=10)

        # Goal
        gx, gy = self.goal_pos
        pygame.draw.rect(screen, self._GOAL, pygame.Rect(gy * cs + 2, gx * cs + 2, cs - 4, cs - 4), border_radius=8)

        # Agent
        ax, ay = self.agent_pos
        center = (ay * cs + cs // 2, ax * cs + cs // 2)
        pygame.draw.circle(screen, self._AGENT, center, cs // 3)


def run_rgb_array_test():
    # Use dummy driver so this works on headless machines too
    prev_driver = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    print("\n=== RGB_ARRAY TEST ===")
    env = GridWorldEnv(grid_size=6, render_mode="rgb_array", cell_size=72, pit_death_reward=-1.0)
    obs, info = env.reset()

    frames = []
    for t in range(10):
        # Step and fetch pixels (either from info['pixels'] if you return it, or call render())
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frame = info.get("pixels", None)
        if frame is None:
            frame = env.render()

        assert isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3, \
            f"Expected (H,W,3) array, got {type(frame)} shape={getattr(frame, 'shape', None)}"
        if frame.dtype != np.uint8:
            # Ensure uint8 for saving/consumers
            frame = frame.astype(np.uint8, copy=False)

        if t == 0:
            print(f"Frame[0] OK: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
        frames.append(frame)

        if terminated or truncated:
            break

    # Save the first frame using pygame (works without extra deps)
    try:
        surf = pygame.surfarray.make_surface(frames[0].transpose(1, 0, 2))  # (W,H,3)
        pygame.image.save(surf, "rgb_frame0.png")
        print("Saved rgb_frame0.png")
    except Exception as e:
        print(f"Could not save frame (non-fatal): {e}")

    env.close()

    # Restore previous driver
    if prev_driver is None:
        os.environ.pop("SDL_VIDEODRIVER", None)
    else:
        os.environ["SDL_VIDEODRIVER"] = prev_driver


def run_human_test():
    print("\n=== HUMAN RENDER TEST ===")
    try:
        env = GridWorldEnv(grid_size=6, render_mode="human", cell_size=72, pit_death_reward=-1.0)
        obs, info = env.reset()
    except Exception as e:
        print(f"Skipping human mode (no display?): {e}")
        return

    total_reward = 0.0
    steps = 0
    MAX_STEPS = 80  # auto-quit after a short demo

    while True:
        # Allow closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                print("Human window closed.")
                return

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated or steps >= MAX_STEPS:
            print(f"Episode finished. Steps={steps} Total reward={total_reward:.2f}")
            break

    env.close()

if __name__ == "__main__":
    run_rgb_array_test()
    run_human_test()
    print("\nDone.")