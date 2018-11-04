import numpy as np
import pygame


class Pygame():
    def __init__(self, caption_name):
        self.dataset = np.empty((0, 2), dtype='f')
        self.radius = 2
        self.color = (0, 0, 255)
        self.thickness = 0
        self.bg_color = (255, 255, 255)
        self.width = 640
        self.height = 480

        self.caption_name = caption_name

    def get_data(self):
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.caption_name)

        running = True
        pushing = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pushing = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    pushing = False

            if pushing and np.random.uniform(0, 1) > .5:
                self.createData(pygame.mouse.get_pos())

            screen.fill(self.bg_color)

            for i, data in enumerate(self.dataset):
                pygame.draw.circle(screen, self.color, (int(data[0]), int(data[1])), self.radius, self.thickness)

            pygame.display.flip()

        pygame.quit()

        return self.dataset

    def createData(self, position):
        (x, y) = position
        r = np.random.uniform(0, 30)
        phi = np.random.uniform(0, 2 * np.pi)
        coord = [x + r * np.cos(phi), y + r * np.sin(phi)]
        global dataset
        self.dataset = np.append(self.dataset, [coord], axis=0)
