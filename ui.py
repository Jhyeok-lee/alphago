import pygame
import pickle
import time
from pygame.locals import *
from pygame.compat import geterror
HIGHT = 6
WIDTH = 6
R = 20
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BOARD_SIZE = (HIGHT*R*5, HIGHT*R*5)
MID = int(HIGHT*R*5/3)
gibo = []
with open('data/gibo.pickle', 'rb') as f:
    gibo = pickle.load(f)
pygame.init()
pygame.display.set_caption('aa')
screen = pygame.display.set_mode(BOARD_SIZE, 0, 32)
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill((211,211,211))
screen.blit(background, (0, 0))
pygame.display.flip()
player = BLACK
i = 0
going = True
while going:
    for event in pygame.event.get():
        print(event)
        print(gibo)
        if event.type == QUIT:
            going = False
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            going = False
        elif event.type == KEYDOWN:
            a = gibo[i]
            r = a // WIDTH
            c = a % WIDTH
            pos = (r*R*2 + MID, c*R*2 + MID)
            pygame.draw.circle(screen, player, pos, R, 0)
            pygame.display.update()
            if player == BLACK:
                player = WHITE
            else:
                player = BLACK
            i += 1
        elif event.type == MOUSEBUTTONDOWN:
            pass
        elif event.type == MOUSEBUTTONUP:
            pass

pygame.quit()