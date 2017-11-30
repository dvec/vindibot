import json
import os

from sys import stdout
from numpy import array, zeros

_POSSIBLE_TILES = \
    {'##': -1, '  ': 0, '$-': 1, '$1': 2, '$2': 3, '$3': 4, '$4': 5, '@1': 6, '@2': 7, '@3': 8, '@4': 9, '[]': 10}

_POSSIBLE_DIRS = {'Stay': 0, 'North': 1, 'South': 2, 'East': 3, 'West': 4}
_DATASET_NORMAL_SIZE = 1200


def load(path, verbose=1):
    if verbose:
        print('Started load function')

    data = []
    for player in os.walk(path):
        for game in player[2]:
            if verbose:
                stdout.write('\rReading file: {}'
                             .format(os.path.join(player[0], game)))

            game_data = []
            with open(os.path.join(player[0], game)) as text:
                for line in text.readlines():
                    line = line[line.find('{'):].strip()
                    if line != '':
                        game_data.append(json.loads(line))
            data.append(game_data)

            for i, game_data in enumerate(data):
                if len(game_data) != _DATASET_NORMAL_SIZE:
                    continue

                x = zeros(shape=[len(game_data), 820])
                y = zeros(shape=[len(game_data), 5])
                for j, step in enumerate(game_data):
                    if bool(step['finished']):
                        break

                    turns_left = step['maxTurns'] - step['turn']
                    heroes = []

                    for hero in step['heroes']:
                        last_dir = _POSSIBLE_DIRS.get(hero.get('lastDir'))
                        heroes.extend((int(hero['elo']), int(hero['pos']['x']), int(hero['pos']['y']),
                                       last_dir if last_dir is not None else 0, int(hero['life']), int(hero['gold']),
                                       int(hero['mineCount']), int(hero['spawnPos']['x']), int(hero['spawnPos']['y'])))

                    board = [-1] * (28 ** 2 - step['board']['size'] ** 2)

                    for index in range(2, step['board']['size'] ** 2 * 2, 2):
                        tile = step['board']['tiles'][index - 2] + step['board']['tiles'][index - 1]
                        board.append(_POSSIBLE_TILES[tile])
                    x.put(j, array(heroes + [turns_left] + board))

                    expected_output = [0] * 5
                    expected_output[heroes[3]] = 1
                    y.put(j, array(expected_output))

                yield x.reshape([1200, 820]), y.reshape([1200, 5])
