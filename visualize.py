import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pygame.locals import *
import pygame
pygame.init()

scale=10
font=pygame.font.SysFont(None,26)
WIDTH = 42*scale
HEIGHT = 42*scale
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0,0,0)
game_display = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('SwimmerWorldv1')
clock = pygame.time.Clock()

def draw_environment(_traj):
    game_display.fill(WHITE)
    #heading=font.render('Episode {}'.format(episode),True,BLACK)
    #game_display.blit(heading,[WIDTH/2 - 30 ,35])
    pygame.draw.circle(game_display, (128,0,128), [int(_traj[0]*scale), int(_traj[1]*scale)], 12)
    pygame.draw.circle(game_display, (0,255,0), [int(_traj[2]*scale), int(_traj[3]*scale)], 3)
    pygame.draw.rect(game_display, BLUE, (30*scale,30*scale, 6*scale, 6*scale))
    pygame.draw.rect(game_display, RED, (6*scale, 6*scale, 30*scale, 30*scale),3)
    pygame.display.update()


def render(traj,framerate=240):
    i=0
    while i<len(traj):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        draw_environment(traj[i])
        i+=1
        clock.tick(framerate)
