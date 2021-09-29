import collections
import random
from enum import Enum

import numpy as np

Binahayat = 100000000

DEBUG = False


def bfs(score_grid, start, goal, avoid_number, avoids=[]):
    height, width = score_grid.shape
    queue = collections.deque([[start]])
    time_out = height * 1000
    seen = set([start])
    i = 0
    while queue:
        if i > time_out:
            return None
        path = queue.popleft()
        y, x = path[-1]
        if (y, x) == goal:
            return path
        near_cells = []
        for y2, x2 in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= x2 < width and 0 <= y2 < height and (y2, x2) not in seen and score_grid[y2][x2] > avoid_number and (
                    y2, x2) not in avoids:
                queue.append(path + [(y2, x2)])
                seen.add((y2, x2))

        i += 1


class Logger:
    outs = []
    inputs = []
    logs = []

    @classmethod
    def log(cls):
        if not DEBUG:
            return
        with open("custom_log_log.text", "w") as f:
            f.writelines([str(l) + "\n" for l in cls.logs])

        with open("custom_log_inputs.text", "w") as f:
            f.write(str(cls.inputs))

        with open("custom_log_outs.text", "w") as f:
            f.writelines([str(o) + "\n" for o in cls.outs])

    @classmethod
    def out(cls, data, write=False):

        if write:
            print(data)
            if DEBUG:
                cls.outs.append(str(data))
        else:
            if DEBUG:
                cls.logs.append(str(data))


class GameInfo:
    vision_range = 0
    bomb_delay = 0
    max_bomb_range = 0
    dead_zone_starting_step = 0
    max_step = 0
    step_count = 0

    @classmethod
    def init(cls, vision_range, bomb_delay, max_bomb_range, dead_zone_starting_step, max_step):
        cls.vision_range = vision_range
        cls.bomb_delay = bomb_delay
        cls.max_bomb_range = max_bomb_range
        cls.dead_zone_starting_step = dead_zone_starting_step
        cls.max_step = max_step
        cls.step_count = 0


class Score:
    class TileScore:
        bomb_score = -Binahayat
        bomb_distance_coefficient = -20
        walked_score = - GameInfo.step_count / 100
        dead_zone_score = - 5000
        trap_score = -4000
        health_upgrade_score = 80
        trap_upgrade_score = 50
        bomb_upgrade_score = 10
        far_center_coefficient = -GameInfo.step_count / 8 * 0.1
        far_upgrade_tile_coefficient = - (150 - GameInfo.step_count)
        stay_score = - GameInfo.step_count / 100
        latest_tile_coefficient = 1
        count_tile_walked_coefficient = - 1.1
        next_bfs_score = 20

    class BombScore:
        box_coefficient = 2
        enemy_coefficient = 0.7

    class Trap:
        enemy_coefficient = 4

    class Random:
        random_score = 0.6

    @classmethod
    def update_scores(cls):
        cls.TileScore.bomb_score = -Binahayat
        cls.TileScore.bomb_distance_coefficient = -20
        cls.TileScore.walked_score = - (GameInfo.step_count + 4) / GameInfo.max_step
        cls.TileScore.dead_zone_score = - 1000
        cls.TileScore.trap_score = -1000
        cls.TileScore.health_upgrade_score = 130
        cls.TileScore.trap_upgrade_score = 100
        cls.TileScore.bomb_upgrade_score = 70
        cls.TileScore.far_center_coefficient = - GameInfo.step_count / GameInfo.max_step
        cls.TileScore.far_upgrade_tile_coefficient = - (GameInfo.max_step - GameInfo.step_count) * 0.9
        cls.TileScore.stay_score = - GameInfo.step_count / GameInfo.max_step
        cls.BombScore.box_coefficient = 4
        cls.BombScore.enemy_coefficient = 2
        cls.TileScore.next_bfs_score = 30


class Tile:
    class TileState(Enum):
        DEAD_ZONE = 0
        FIRE_SIDE = 1
        BOX = 2
        WALL = 3
        BOMB = 4
        BOMB_RANGE_UPGRADE = 5
        HEALTH_UPGRADE = 6
        TRAP_UPGRADE = 7
        PLAYER = 8

    class TileType(Enum):
        EMPTY = 0
        WALL = 1
        BOX = 2
        BOMB = 3
        TRAP = 4
        BOMB_RANGE_UPGRADE = 5
        HEALTH_UPGRADE = 6
        TRAP_UPGRADE = 7
        UnKnown = 8

    def __init__(self, x, y, state_number, tile_type=TileType.UnKnown, info=None):
        self.x = x
        self.y = y
        self.type = tile_type
        self.state_number = state_number
        self.info = {} if info is None else info

    def is_bomb(self):
        return self.is_special_state(state=self.TileState.BOMB)

    def is_wall(self):
        return self.is_special_state(state=self.TileState.WALL)

    def is_box(self):
        return self.is_special_state(state=self.TileState.BOX)

    def is_trap(self):
        return self.type == self.TileType.TRAP

    def is_unknown(self):
        return self.type == self.TileType.UnKnown

    def is_special_state(self, state: TileState.BOMB):
        bits = list(bin(self.state_number)[2:].zfill(9))
        bits.reverse()
        if int(bits[state.value]) == 1:
            return True
        else:
            return False

    def is_in_states(self, states: list):
        bits = list(bin(self.state_number)[2:].zfill(9))
        bits.reverse()
        for state in states:
            if int(bits[state.value]) == 1:
                return True

        return False

    @classmethod
    def state_number_to_type(cls, state_number):
        bits = list(bin(state_number)[2:].zfill(9))

        bits.reverse()

        if int(bits[cls.TileState.WALL.value]) == 1:
            return cls.TileType.WALL

        if int(bits[cls.TileState.BOX.value]) == 1:
            return cls.TileType.BOX

        if int(bits[cls.TileState.BOMB.value]) == 1:
            return cls.TileType.BOMB

        if int(bits[cls.TileState.TRAP_UPGRADE.value]) == 1:
            return cls.TileType.TRAP_UPGRADE
        if int(bits[cls.TileState.BOMB_RANGE_UPGRADE.value]) == 1:
            return cls.TileType.BOMB_RANGE_UPGRADE

        if int(bits[cls.TileState.HEALTH_UPGRADE.value]) == 1:
            return cls.TileType.HEALTH_UPGRADE

        else:
            return cls.TileType.EMPTY

    def get_address(self):
        return (self.y, self.x)

    def __repr__(self):
        if self.is_special_state(state=self.TileState.PLAYER):
            return "P"

        if self.type == self.TileType.EMPTY:
            return "O"

        if self.is_wall():
            return "*"
        if self.is_box():
            return "+"
        if self.is_bomb():
            return "B"

        if self.is_special_state(state=self.TileState.BOMB_RANGE_UPGRADE):
            return "R"
        if self.is_special_state(state=self.TileState.TRAP_UPGRADE):
            return "T"
        if self.is_special_state(state=self.TileState.HEALTH_UPGRADE):
            return "H"
        if self.type == self.TileType.UnKnown:
            return "N"
        if self.type == self.TileType.TRAP:
            return "X"

        return "F"


class Map:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.tiles = np.array(
            [[Tile(x=x, y=y, state_number=0, tile_type=Tile.TileType.UnKnown) for x in range(width)] for y in
             range(height)])

        self.scores = np.zeros((height, width))
        self.is_deadzone_time = False

    def get_boxes(self):
        l = []
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[y][x]
                tile: Tile
                if tile.is_special_state(state=Tile.TileState.BOX):
                    l.append(tile)

        return l

    def get_traps(self):
        l = []
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[y][x]
                tile: Tile
                if tile.is_trap():
                    l.append(tile)

        return l

    def update_scores(self, bomb_range, current_tile, escape_enemy=None):
        bomb_sides = self.get_bomb_range_side(bomb_range=bomb_range)
        if GameInfo.step_count >= GameInfo.dead_zone_starting_step:
            self.is_deadzone_time = True

        for y in range(self.height):
            for x in range(self.width):
                score = 0
                tile = self.tiles[y][x]
                tile: Tile

                if tile.is_special_state(state=Tile.TileState.DEAD_ZONE):
                    self.is_deadzone_time = True
                    score -= Binahayat
                if tile.is_special_state(state=Tile.TileState.WALL):
                    score -= Binahayat

                if tile in bomb_sides:
                    score -= 60
                if tile.is_trap():
                    score -= 60

                if GameInfo.step_count <= GameInfo.dead_zone_starting_step / 2:
                    if tile.is_special_state(state=Tile.TileState.HEALTH_UPGRADE):
                        score += 200

                    if tile.is_special_state(state=Tile.TileState.TRAP_UPGRADE):
                        score += 120
                    if tile.is_special_state(state=Tile.TileState.BOMB_RANGE_UPGRADE):
                        score += 40
                if tile.is_trap():
                    score -= 1600

                if GameInfo.step_count <= (2 * GameInfo.dead_zone_starting_step) / 3:
                    if tile.is_special_state(state=Tile.TileState.HEALTH_UPGRADE):
                        score += 30
                    if tile.is_special_state(state=Tile.TileState.TRAP_UPGRADE):
                        score += 18

                elif GameInfo.step_count <= GameInfo.dead_zone_starting_step:
                    if tile.is_special_state(state=Tile.TileState.HEALTH_UPGRADE):
                        score += 7
                    if tile.is_special_state(state=Tile.TileState.TRAP_UPGRADE):
                        score += 3

                score -= self.manhatan_distance(tile, self.get_center_tile()) * (
                        GameInfo.step_count / GameInfo.dead_zone_starting_step) * 2

                if GameInfo.step_count <= GameInfo.dead_zone_starting_step / 2:
                    if tile.is_special_state(state=Tile.TileState.BOX):
                        score += 30
                    for n_tile in self.get_neighbour_tiles(tile=tile):
                        n_tile: Tile
                        if not n_tile.is_special_state(state=Tile.TileState.WALL):
                            score -= 1

                else:
                    if tile.is_special_state(state=Tile.TileState.BOX):
                        score -= 1000

                if escape_enemy is not None and not self.is_deadzone_time:
                    if escape_enemy:
                        if tile.is_special_state(state=Tile.TileState.PLAYER) and tile != current_tile:
                            score -= 60
                    else:
                        if tile.is_special_state(state=Tile.TileState.PLAYER) and tile != current_tile:
                            score += 3

                else:
                    if tile.is_special_state(state=Tile.TileState.PLAYER) and tile != current_tile:
                        score -= 10

                if self.is_deadzone_time:
                    score -= self.manhatan_distance(tile, self.get_center_tile()) * 3
                    if self.width <= self.height:
                        score -= abs(tile.x - self.width / 2) * 5
                    else:
                        score -= abs(tile.y - self.height / 2) * 5

                score -= self.manhatan_distance(tile, current_tile) * 0.1
                score += random.random() * 0.4

                self.scores[y][x] = score

    def get_tiles_in_manhatan(self, tile, distance):
        l = []
        for y in range(self.height):
            for x in range(self.width):
                new_tile = self.get_tile(x=x, y=y)
                if self.manhatan_distance(tile, new_tile) <= distance:
                    l.append(new_tile)

        return l

    def update_map_with_vision(self, vision_tiles):
        for tile_tuple in vision_tiles:
            (y, x), state_number, tile_type = tile_tuple
            tile = self.get_tile(x=x, y=y)
            tile.state_number = state_number
            if not tile.is_bomb() and tile_type == Tile.TileType.BOMB:
                tile.info["time"] = GameInfo.bomb_delay
            if not tile.is_trap():
                tile.type = tile_type

    def update_bombs_time(self):
        for x in range(self.width):
            for y in range(self.height):
                tile = self.tiles[y][x]
                if tile.is_bomb():
                    if "time" in tile.info.keys():
                        time = tile.info["time"] - 1
                        tile.info["time"] = time
                    else:
                        time = GameInfo.bomb_delay
                        tile.info["time"] = time
                    if time < 1:
                        # TODO tile_state must be changed
                        tile.type = Tile.TileType.EMPTY
                        tile.info = {}

    def update_map(self, vision_tiles):
        self.update_map_with_vision(vision_tiles=vision_tiles)
        self.update_bombs_time()

    def get_neighbour_tiles(self, tile):
        current_tiles = []
        x = tile.x
        y = tile.y

        for (y1, x1) in [(y - 1, x), (y, x - 1), (y, x), (y, x + 1), (y + 1, x)]:
            if x1 in range(0, self.width) and y1 in range(0, self.height):
                tile = self.tiles[y1][x1]
                current_tiles.append(tile)

        return current_tiles

    def get_bomb_tiles(self):
        bomb_lists = []
        for x in range(self.width):
            for y in range(self.height):
                tile = self.tiles[y][x]
                if tile.is_bomb():
                    bomb_lists.append(tile)

        return bomb_lists

    def manhatan_distance(self, tile1, tile2):
        return abs(tile1.x - tile2.x) + abs(tile1.y - tile2.y)

    def get_nearest_bomb(self, tile):
        l = self.get_bomb_tiles()
        if len(l) > 0:
            return min(l, key=lambda item: self.manhatan_distance(tile1=tile, tile2=item))
        return None

    def get_nearest_upgrade(self, tile):  # TODO kind of upgrade
        bomb_lists = []
        for x in range(self.width):
            for y in range(self.height):
                tile = self.tiles[y][x]
                if tile.is_bomb():
                    bomb_lists.append(tile)
        l = bomb_lists
        if len(l) > 0:
            return min(l, key=lambda item: self.manhatan_distance(tile1=tile, tile2=item))
        return None

    def get_tile(self, x, y):
        if x in range(0, self.width) and y in range(0, self.height):
            return self.tiles[y][x]
        return None

    def get_center_tile(self):
        return self.get_tile(x=self.width // 2, y=self.height // 2)

    def get_bomb_one_range_sides(self, tile: Tile, bomb_range):
        l = []
        x, y = tile.x, tile.y
        for i in range(bomb_range + 1):
            try:
                new_tile = self.tiles[y][x + i]
                l.append(new_tile)
                if new_tile.is_special_state(Tile.TileState.WALL) or new_tile.is_special_state(Tile.TileState.BOX):
                    break

            except:
                pass

        for i in range(bomb_range + 1):
            try:
                new_tile = self.tiles[y][x - i]
                l.append(new_tile)
                if new_tile.is_special_state(Tile.TileState.WALL) or new_tile.is_special_state(Tile.TileState.BOX):
                    break

            except:
                pass
        for i in range(bomb_range + 1):
            try:
                new_tile = self.tiles[y + i][x]
                l.append(new_tile)
                if new_tile.is_special_state(Tile.TileState.WALL) or new_tile.is_special_state(Tile.TileState.BOX):
                    break

            except:
                pass
        for i in range(bomb_range + 1):
            try:
                new_tile = self.tiles[y - i][x]
                l.append(new_tile)
                if new_tile.is_special_state(Tile.TileState.WALL) or new_tile.is_special_state(Tile.TileState.BOX):
                    break

            except:
                pass

        return l

    def get_bomb_range_side(self, bomb_range):
        l = []
        for bomb_tile in self.get_bomb_tiles():
            l.extend(
                self.get_bomb_one_range_sides(tile=bomb_tile,
                                              bomb_range=bomb_range))

        return l

    def get_tile_score(self, tile):
        y, x = tile.y, tile.x
        return self.scores[y][x]


class Player:
    def __init__(self, tile, health):
        self.tile = tile
        self.health = health


class MyPlayer(Player):
    def __init__(self, tile, health, bomb_range, trap_count):
        super(MyPlayer, self).__init__(tile=tile, health=health)
        self.bomb_range = bomb_range
        self.trap_count = trap_count
        self.last_action_played = None
        self.health_upgrade_count = 0
        self.vision_tile_tuples = []
        self.latest_tile = None


class AI:
    class Action(Enum):
        GO_LEFT = 0
        GO_RIGHT = 1
        GO_UP = 2
        GO_DOWN = 3
        STAY = 4
        PLACE_BOMB = 5
        PLACE_TRAP_LEFT = 6
        PLACE_TRAP_RIGHT = 7
        PLACE_TRAP_UP = 8
        PLACE_TRAP_DOWN = 9
        INIT = 10
        NO_ACTION = 11

        @classmethod
        def get_action(cls, number):
            return [
                cls.GO_LEFT,
                cls.GO_RIGHT,
                cls.GO_UP,
                cls.GO_DOWN,
                cls.STAY,
                cls.PLACE_BOMB,
                cls.PLACE_TRAP_LEFT,
                cls.PLACE_TRAP_RIGHT,
                cls.PLACE_TRAP_UP,
                cls.PLACE_TRAP_DOWN,
                cls.INIT,
                cls.NO_ACTION,
            ][number]

    def __init__(self, game_map: Map, my_player: MyPlayer):
        self.game_map = game_map
        self.my_player = my_player
        self.enemy = Player(tile=None, health=None)
        self.latest_walked_tiles = []
        self.history_walked_tiles = []
        self.bomb_time = 0
        self.bomb_tile = None
        self.latest_wanted_action = None
        self.escape_enemy = None

    @property
    def visible_tiles(self):
        return [self.game_map.get_tile(x=item[0][1], y=item[0][0]) for item in self.my_player.vision_tile_tuples]

    def score_for_bomb_side(self, tile):
        if tile.is_special_state(state=Tile.TileState.BOMB):
            return Score.TileScore.bomb_score

        bomb_tile = self.game_map.get_nearest_bomb(tile=tile)
        if bomb_tile is None:
            return 0
        score = 0

        bomb_distance_score = (GameInfo.max_bomb_range - self.game_map.manhatan_distance(tile1=tile, tile2=bomb_tile))
        if bomb_distance_score > 0:
            score += bomb_distance_score

        for (point, bomb_range) in [(300, self.my_player.bomb_range), (10, self.my_player.bomb_range + 1),
                                    (3, self.my_player.bomb_range + 2)]:

            bomb_tiles = self.game_map.get_bomb_range_side(bomb_range=bomb_range)
            if tile in bomb_tiles:
                score -= point

        return score

    def tile_score(self, tile: Tile):
        score = 0
        if tile.type in [Tile.TileType.WALL, Tile.TileType.BOX]:
            return -Binahayat
        if tile.is_bomb():
            return -(Binahayat - 100)
        n_tiles = self.game_map.get_neighbour_tiles(tile=tile)

        if tile.type == Tile.TileType.TRAP:
            score += Score.TileScore.trap_score

        if tile.is_special_state(state=Tile.TileState.FIRE_SIDE):
            score -= 3000

        for n_tile in n_tiles:
            if n_tile.is_special_state(state=Tile.TileState.FIRE_SIDE):
                score -= 500
                break

        if tile.is_special_state(state=Tile.TileState.DEAD_ZONE):
            score += Score.TileScore.dead_zone_score

        fire_score = self.score_for_bomb_side(tile=tile)
        score += fire_score

        if tile.type == Tile.TileType.HEALTH_UPGRADE:
            score += Score.TileScore.health_upgrade_score

        if tile.type == Tile.TileType.TRAP_UPGRADE:
            score += Score.TileScore.trap_upgrade_score

        if tile.type == Tile.TileType.BOMB_RANGE_UPGRADE:
            score += Score.TileScore.bomb_upgrade_score

        if self.escape_enemy is not None:

            for n_tile in n_tiles:
                if n_tile.is_special_state(state=Tile.TileState.PLAYER) and n_tile != self.my_player.tile:
                    if self.escape_enemy:
                        score -= 50
                    else:
                        score += 10

        return score

    def tile_is_enemy(self, tile):
        if tile.is_special_state(state=Tile.TileState.PLAYER) and tile != self.my_player.tile:
            return True
        return False

    def stay_score(self):
        current_tile = self.my_player.tile
        if current_tile.is_special_state(state=Tile.TileState.BOMB):
            return -Binahayat
        return self.tile_score(tile=current_tile)

    def go_up_score(self):
        current_tile = self.my_player.tile
        tile = self.game_map.get_tile(x=current_tile.x, y=current_tile.y - 1)
        if tile is None or self.tile_is_enemy(tile=tile):
            return -Binahayat
        return self.tile_score(tile=tile)

    def go_down_score(self):
        # return 0
        current_tile = self.my_player.tile
        tile = self.game_map.get_tile(x=current_tile.x, y=current_tile.y + 1)
        if tile is None or self.tile_is_enemy(tile=tile):
            return -Binahayat

        return self.tile_score(tile=tile)

    def go_left_score(self):
        # return 0
        current_tile = self.my_player.tile
        tile = self.game_map.get_tile(x=current_tile.x - 1, y=current_tile.y)
        if tile is None or self.tile_is_enemy(tile=tile):
            return -Binahayat
        return self.tile_score(tile=tile)

    def go_right_score(self):
        # return 0
        current_tile = self.my_player.tile
        tile = self.game_map.get_tile(x=current_tile.x + 1, y=current_tile.y)
        if tile is None or self.tile_is_enemy(tile=tile):
            return -Binahayat
        return self.tile_score(tile=tile)

    # def stay_score(self):
    #     current_tile = self.my_player.tile
    #     return self.game_map.get_tile_score(current_tile)
    #
    # def go_up_score(self):
    #     current_tile = self.my_player.tile
    #     tile = self.game_map.get_tile(x=current_tile.x, y=current_tile.y - 1)
    #     if tile is None:
    #         return -Binahayat
    #
    #     return self.game_map.get_tile_score(tile)
    #
    # def go_down_score(self):
    #     current_tile = self.my_player.tile
    #     tile = self.game_map.get_tile(x=current_tile.x, y=current_tile.y + 1)
    #     if tile is None:
    #         return -Binahayat
    #     return self.game_map.get_tile_score(tile)
    #
    # def go_left_score(self):
    #     current_tile = self.my_player.tile
    #     tile = self.game_map.get_tile(x=current_tile.x - 1, y=current_tile.y)
    #     if tile is None:
    #         return -Binahayat
    #     return self.game_map.get_tile_score(tile)
    #
    # def go_right_score(self):
    #     current_tile = self.my_player.tile
    #     tile = self.game_map.get_tile(x=current_tile.x + 1, y=current_tile.y)
    #     if tile is None:
    #         return -Binahayat
    #     return self.game_map.get_tile_score(tile)

    def place_bomb_score(self):

        current_tile = self.my_player.tile
        if current_tile.is_special_state(state=Tile.TileState.BOMB):
            return -Binahayat

        score = self.stay_score() - 0.5

        fire_side_tiles = self.game_map.get_bomb_one_range_sides(tile=self.my_player.tile,
                                                                 bomb_range=self.my_player.bomb_range)
        for fire_tile in fire_side_tiles:
            if fire_tile.is_special_state(Tile.TileState.BOX):
                score += 70
                if fire_tile.is_special_state(Tile.TileState.BOMB):
                    score -= Binahayat

        for tile in self.visible_tiles:
            if tile.is_special_state(state=Tile.TileState.PLAYER) and tile != current_tile and tile in fire_side_tiles:
                if self.enemy is not None and self.enemy.health is not None:
                    if self.my_player.health >= self.enemy.health:
                        score += random.random() * 18

                score += random.random() * 3
                score -= self.game_map.manhatan_distance(tile, current_tile) * 0.2

        score = score

        if self.bomb_time > 0:
            score -= 300
        manhatan_tiles = self.game_map.get_tiles_in_manhatan(tile=self.my_player.tile,
                                                             distance=GameInfo.bomb_delay)

        if self.game_map.is_deadzone_time:
            score -= 300

        for tile in manhatan_tiles:
            tile: Tile
            if tile not in fire_side_tiles and not tile.is_in_states(
                    states=[Tile.TileState.WALL, Tile.TileState.BOX, Tile.TileState.PLAYER,
                            Tile.TileState.DEAD_ZONE]):
                path = bfs(self.game_map.scores, current_tile.get_address(),
                           tile.get_address(),
                           -500, avoids=[])
                if path is not None and len(path) < GameInfo.bomb_delay / 2:
                    # if path is not None :
                    return score

        return -5000

    def place_trap_score(self, tile, current_tile):

        if tile is None:
            return -Binahayat
        if tile.is_in_states(states=[Tile.TileState.WALL, Tile.TileState.BOX]):
            return -Binahayat

        if tile.type == Tile.TileType.TRAP:
            return -Binahayat
        if self.my_player.trap_count == 0:
            return -Binahayat

        score = self.stay_score()
        if tile.is_special_state(state=Tile.TileState.PLAYER) and tile != current_tile:
            return Binahayat

        neighbours = self.game_map.get_neighbour_tiles(tile=current_tile)
        for n_tile in neighbours:
            if n_tile.is_trap():
                return -Binahayat
        neighbours = self.game_map.get_neighbour_tiles(tile=tile)

        path = bfs(score_grid=self.game_map.scores, start=current_tile.get_address(),
                   goal=self.game_map.get_center_tile().get_address(), avoid_number=-500, avoids=[tile.get_address()])
        if path is None:
            return -Binahayat

        if self.my_player.trap_count > 2:
            for n_tile in neighbours:
                if n_tile.is_special_state(state=Tile.TileState.PLAYER) and n_tile != current_tile:
                    return 5000

        # for tile in self.vis
        if self.my_player.trap_count > 3 or (self.my_player.trap_count > 2 and self.game_map.is_deadzone_time):
            for v_tile in self.visible_tiles:
                if v_tile.is_special_state(state=Tile.TileState.PLAYER) and tile != current_tile:
                    if self.game_map.manhatan_distance(tile, v_tile) < self.game_map.manhatan_distance(v_tile,
                                                                                                       current_tile):
                        if self.game_map.is_deadzone_time:
                            return 80  # TODO near enemy

                        return 700  # TODO near enemy

        return -Binahayat

    def place_trap_up(self):
        current_tile = self.my_player.tile
        tile = self.game_map.get_tile(x=current_tile.x, y=current_tile.y - 1)
        score = self.place_trap_score(tile=tile, current_tile=current_tile)
        return score

    def place_trap_down(self):
        current_tile = self.my_player.tile
        tile = self.game_map.get_tile(x=current_tile.x, y=current_tile.y + 1)
        score = self.place_trap_score(tile=tile, current_tile=current_tile)
        return score

    def place_trap_right(self):
        current_tile = self.my_player.tile
        tile = self.game_map.get_tile(x=current_tile.x + 1, y=current_tile.y)
        score = self.place_trap_score(tile=tile, current_tile=current_tile)
        return score

    def place_trap_left(self):
        current_tile = self.my_player.tile
        tile = self.game_map.get_tile(x=current_tile.x - 1, y=current_tile.y)
        score = self.place_trap_score(tile=tile, current_tile=current_tile)
        return score

    def get_goal(self):
        if self.bomb_time > 0:
            bomb_tile = self.bomb_tile
            if bomb_tile is None:
                bomb_tile = self.game_map.get_nearest_bomb(tile=self.my_player.tile)
            if bomb_tile is not None:
                bomb_sides = self.game_map.get_bomb_one_range_sides(tile=bomb_tile,
                                                                    bomb_range=self.my_player.bomb_range)

                manhatan_tiles = self.game_map.get_tiles_in_manhatan(tile=bomb_tile, distance=GameInfo.bomb_delay)
                manhatan_tiles = sorted(manhatan_tiles, key=lambda item: self.game_map.manhatan_distance(item,
                                                                                                         self.game_map.get_center_tile()))
                avoid_tiles = self.game_map.get_boxes()
                avoid_tiles.extend(self.game_map.get_traps())
                avoids = [tile.get_address() for tile in avoid_tiles]
                for tile in manhatan_tiles:
                    if tile not in bomb_sides:
                        tile: Tile
                        if not tile.is_unknown():
                            if not tile.is_in_states(
                                    states=[Tile.TileState.WALL, Tile.TileState.BOX, Tile.TileState.DEAD_ZONE]):
                                path = bfs(self.game_map.scores, self.my_player.tile.get_address(), tile.get_address(),
                                           avoid_number=-Binahayat, avoids=avoids)
                                if path is not None and len(path) < GameInfo.bomb_delay / 2:
                                    return tile

        index = self.game_map.scores.argmax()
        Y = index // self.game_map.width
        X = index % self.game_map.width

        return self.game_map.get_tile(X, Y)

    def get_by_path_score(self, next_tile):
        current_tile = self.my_player.tile
        score = Score.TileScore.next_bfs_score
        if next_tile is None:
            return [0, 0, 0, 0, 0]

        if next_tile == current_tile:
            return [score, 0, 0, 0, 0]

        if next_tile.y < current_tile.y:
            return [0, score, 0, 0, 0]

        if next_tile.x > current_tile.x:
            return [0, 0, score, 0, 0]

        if next_tile.y > current_tile.y:
            return [0, 0, 0, score, 0]

        if next_tile.x < current_tile.x:
            return [0, 0, 0, 0, score]

    def get_score_actions(self):
        #
        # for walked_tile in self.latest_walked_tiles:
        #     Y, X = walked_tile.get_address()
        #     self.game_map.scores[Y][X] = self.game_map.scores[Y][X] - 0.1

        goal_tile = self.get_goal()
        current_tile = self.my_player.tile
        box_avoids = [tile.get_address() for tile in self.game_map.get_boxes()]
        trap_avoids = [tile.get_address() for tile in self.game_map.get_traps()]

        path = bfs(self.game_map.scores, start=current_tile.get_address(), goal=goal_tile.get_address(),
                   avoid_number=-1000, avoids=box_avoids + trap_avoids)
        if path is None:
            bfs(self.game_map.scores, start=current_tile.get_address(), goal=goal_tile.get_address(),
                avoid_number=-Binahayat, avoids=trap_avoids)
        if path is None:
            if not goal_tile.is_special_state(state=Tile.TileState.PLAYER):
                bfs(self.game_map.scores, start=current_tile.get_address(), goal=goal_tile.get_address(),
                    avoid_number=-Binahayat, avoids=[])

        next_tile = None

        if path is None:
            next_tile = None
        elif len(path) > 1:
            y, x = path[1]
            next_tile = self.game_map.get_tile(x, y)
        elif len(path) == 1:
            next_tile = current_tile
        next_tile: Tile
        goal_scores = self.get_by_path_score(next_tile=next_tile)
        Logger.out(f"BSCORE : {self.place_bomb_score()}")

        return [

            {
                "action": self.Action.STAY,
                "score": self.stay_score() + random.random() * Score.Random.random_score + goal_scores[0]
            },
            {
                "action": self.Action.GO_UP,
                "score": self.go_up_score() + random.random() * Score.Random.random_score + goal_scores[1]
            },

            {
                "action": self.Action.GO_RIGHT,
                "score": self.go_right_score() + random.random() * Score.Random.random_score + goal_scores[2]
            },
            {
                "action": self.Action.GO_DOWN,
                "score": self.go_down_score() + random.random() * Score.Random.random_score + goal_scores[3]
            },
            {
                "action": self.Action.GO_LEFT,
                "score": self.go_left_score() + random.random() * Score.Random.random_score + goal_scores[4]
            },
            {
                "action": self.Action.PLACE_BOMB,
                "score": self.place_bomb_score() + random.random() * Score.Random.random_score + goal_scores[0]
            },
            {
                "action": self.Action.PLACE_TRAP_UP,
                "score": self.place_trap_up() + random.random() * Score.Random.random_score + goal_scores[0]
            },
            {
                "action": self.Action.PLACE_TRAP_DOWN,
                "score": self.place_trap_down() + random.random() * Score.Random.random_score + goal_scores[0]
            },
            {
                "action": self.Action.PLACE_TRAP_LEFT,
                "score": self.place_trap_left() + random.random() * Score.Random.random_score + goal_scores[0]
            },
            {
                "action": self.Action.PLACE_TRAP_RIGHT,
                "score": self.place_trap_right() + random.random() * Score.Random.random_score + goal_scores[0]
            },
        ]

    def choose_action(self):
        # TODO new choose action
        score_actions = self.get_score_actions()
        Logger.out(score_actions[5])

        if self.latest_wanted_action != self.my_player.last_action_played:
            for action_score in score_actions:
                if action_score["action"] == self.latest_wanted_action:
                    action_score["score"] = -Binahayat
                break

        random.shuffle(score_actions)
        action = max(score_actions, key=lambda item: item["score"])["action"]

        return action
        # return random.choice(
        #     [self.Action.STAY, self.Action.GO_RIGHT, self.Action.GO_LEFT, self.Action.GO_DOWN, self.Action.GO_UP,
        #      self.Action.PLACE_BOMB])

    def update_knowledge(self):
        self.game_map.update_map(self.my_player.vision_tile_tuples)
        if self.my_player.last_action_played == self.Action.PLACE_TRAP_RIGHT:
            x = self.my_player.tile.x + 1
            y = self.my_player.tile.y
            tile = self.game_map.get_tile(x, y)
            if not tile.is_special_state(Tile.TileState.PLAYER):
                tile.type = Tile.TileType.TRAP

        if self.my_player.last_action_played == self.Action.PLACE_TRAP_LEFT:
            x = self.my_player.tile.x - 1
            y = self.my_player.tile.y
            tile = self.game_map.get_tile(x, y)
            if not tile.is_special_state(Tile.TileState.PLAYER):
                tile.type = Tile.TileType.TRAP

        if self.my_player.last_action_played == self.Action.PLACE_TRAP_UP:
            x = self.my_player.tile.x
            y = self.my_player.tile.y - 1
            tile = self.game_map.get_tile(x, y)
            if not tile.is_special_state(Tile.TileState.PLAYER):
                tile.type = Tile.TileType.TRAP
        if self.my_player.last_action_played == self.Action.PLACE_TRAP_DOWN:
            x = self.my_player.tile.x
            y = self.my_player.tile.y + 1
            tile = self.game_map.get_tile(x, y)
            if not tile.is_special_state(Tile.TileState.PLAYER):
                tile.type = Tile.TileType.TRAP

        if self.my_player.last_action_played == self.Action.PLACE_BOMB:
            self.bomb_time = GameInfo.bomb_delay // 2
            self.bomb_tile = self.my_player.tile
        else:
            self.bomb_time -= 1
            if self.bomb_time < 0:
                self.bomb_time = 0
                self.bomb_tile = None

        Score.update_scores()
        escape_enemy = None
        if GameInfo.step_count < (3 * GameInfo.dead_zone_starting_step) / 4:
            escape_enemy = None

        elif self.enemy is not None and self.enemy.health is not None:
            if self.enemy.health < self.my_player.health:
                if self.my_player.trap_count > 0:
                    escape_enemy = False
                else:
                    escape_enemy = True

            elif self.enemy.health == self.my_player.health:
                if self.my_player.trap_count > 1:
                    escape_enemy = False
                elif self.my_player.trap_count == 1:
                    escape_enemy = None
                else:
                    escape_enemy = True

            else:
                escape_enemy = True

        self.escape_enemy = escape_enemy
        self.game_map.update_scores(bomb_range=self.my_player.bomb_range, current_tile=self.my_player.tile,
                                    escape_enemy=escape_enemy)
        Logger.out("_____________________")
        for row in self.game_map.scores.astype(int).tolist():
            # for row in self.game_map.scores.round(1).tolist():
            Logger.out(str(row))

    def turn(self):
        self.update_knowledge()
        action = self.choose_action()
        self.latest_wanted_action = action
        Logger.out(str(self.game_map.tiles))
        return action


class Game:
    def __init__(self):
        data = input()
        # data ="init 11 15 9 13 3 2 1 5 8 5 150 5 400"
        # data = [11, 15, 9, 13, 3, 2, 1, 5, 8, 5, 150, 5, 400]
        init_data = [int(item) for item in data.split(" ")[1:]]
        game_map = Map(height=init_data[0], width=init_data[1])
        tile = Tile(x=init_data[3], y=init_data[2], state_number=0, tile_type=Tile.TileType.EMPTY)
        my_player = MyPlayer(tile=tile, health=init_data[4], bomb_range=init_data[5], trap_count=init_data[6])

        GameInfo.init(vision_range=init_data[7], bomb_delay=init_data[8], max_bomb_range=init_data[9],
                      dead_zone_starting_step=init_data[10], max_step=init_data[12])

        self.ai = AI(game_map=game_map, my_player=my_player)
        print("init confirm")

    def get_info_round_and_update(self):
        state_msg = input()
        if 'term' in state_msg:
            return "finish"
        data = [int(item) for item in state_msg.split(" ")[:-1]]
        GameInfo.step_count = data[0]

        my_player = self.ai.my_player
        my_player.last_action_played = self.ai.Action.get_action(data[1])

        my_player.tile = self.ai.game_map.get_tile(x=data[3], y=data[2])

        my_player.health = data[4]
        my_player.health_upgrade_count = data[5]
        my_player.bomb_range = data[6]
        my_player.trap_count = data[7]

        if data[8] == 1:
            self.ai.enemy.x = data[9]
            self.ai.enemy.y = data[10]
            self.ai.enemy.health = data[11]
            vision_base_index = 12



        else:
            self.ai.enemy.x = None
            self.ai.enemy.y = None
            self.ai.enemy.health = None
            vision_base_index = 9
        vision_tiles = []
        for index in range(vision_base_index + 1, len(data), 3):
            y = data[index]
            x = data[index + 1]
            new_tile = ((y, x), data[index + 2], Tile.state_number_to_type(state_number=data[index + 2]))

            vision_tiles.append(new_tile)

        self.ai.my_player.vision_tile_tuples = vision_tiles

    def choose_action(self, action):
        print(action.value)

    def run(self):
        try:
            while True:
                info = self.get_info_round_and_update()

                if info == "finish":
                    Logger.log()
                    break
                try:
                    action = self.ai.turn()
                    self.choose_action(action)
                except Exception as e:
                    if DEBUG:
                        raise e
                    print(AI.Action.STAY.value)
        except Exception as e:
            Logger.log()
            raise e


if __name__ == '__main__':
    Game().run()
