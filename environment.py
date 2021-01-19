import numpy as np
import random
import curses

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class Environment:

    def __init__(self, fileName, is_slippery=True, slip_prob=0.25):
        self.lake = np.array(self.loadEnvironment(fileName))
        # Environment dimensions
        self.actions = 4  # left, down, right, up
        self.states = self.lake.size
        # Starting position and state
        self.pos = np.argwhere(self.lake == 'S')[0]
        self.state = self.pos[0] * len(self.lake[0]) + self.pos[1]
        # Goal position
        self.goal = np.argwhere(self.lake == 'G')[0]
        # Current reward and status
        self.reward = 0
        self.done = False
        # Terminal view initialization
        self.view = None
        self.is_slippery = is_slippery
        self.slip_prob = slip_prob

    def reset(self):
        self.pos = np.argwhere(self.lake == 'S')[0]
        self.state = self.pos[0] * len(self.lake[0]) + self.pos[1]
        self.reward = 0
        self.done = False
        return self.state

    def initCurses(self):
        self.view = curses.initscr()
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        # Agent color definition
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_RED)
        # Success color definition
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_GREEN)

    def deInitCurses(self):
        curses.endwin()

    def loadEnvironment(self, fileName):
        lake = []
        with open(fileName, "r") as file:
            for line in file:
                lake.append(list(line.rstrip()))
        return lake

    def sampleAction(self):
        action = random.randrange(0, self.actions)
        return action

    def step(self, action):
        if self.is_slippery:
            slip_treshold = random.uniform(0, 1)
            if slip_treshold < self.slip_prob:
                action = self.sampleAction()

        if action == LEFT:
            self.pos[1] = max(self.pos[1] - 1, 0)
        elif action == DOWN:
            self.pos[0] = min(self.pos[0] + 1, len(self.lake)-1)
        elif action == RIGHT:
            self.pos[1] = min(self.pos[1] + 1,  len(self.lake[0])-1)
        else:
            self.pos[0] = max(self.pos[0] - 1, 0)

        # Calculate state number from position cordinates
        self.state = self.pos[0] * len(self.lake[0]) + self.pos[1]

        if self.lake[self.pos[0], self.pos[1]] == 'G':
            self.reward = 1
            self.done = True
        elif self.lake[self.pos[0], self.pos[1]] == 'H':
            self.reward = 0
            self.done = True
        else:
            self.reward = 0
            self.done = False

        return self.state, self.reward, self.done

    def render(self):
        temp = ''
        for row in self.lake:
            temp += ' '.join(row)
            temp += "\n"

        self.view.addstr(0, 0, temp)
        self.view.addstr(self.pos[0], self.pos[1]*2,
                         self.lake[self.pos[0], self.pos[1]], curses.color_pair(1))
        if self.done == True:
            if self.reward == 0:
                self.view.addstr(len(self.lake) + 1, 0, "You failed!", curses.color_pair(1))
            else:
                self.view.addstr(len(self.lake) + 1, 0, "Success!", curses.color_pair(2))
        else:
            self.view.addstr(len(self.lake) + 1, 0, "           ", curses.color_pair(0))
        self.view.refresh()
