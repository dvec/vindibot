import json
import os

from sys import stdout
from numpy import array


class DataReader(object):
    _POSSIBLE_TILES = \
        {'##': -1, '  ': 0, '$-': 1, '$1': 2, '$2': 3, '$3': 4, '$4': 5, '@1': 6, '@2': 7, '@3': 8, '@4': 9, '[]': 10}

    _POSSIBLE_DIRS = {'Stay': 0, 'North': 1, 'South': 2, 'East': 3, 'West': 4}
    _DATASET_NORMAL_SIZE = 1200

    _path = None
    _from = 0
    _games_count = 0

    def __init__(self, path):
        self._path = path
        for i in list(os.walk(self._path)):
            self._games_count += len(i[2])

    def load(self, batch_size=100, verbose=1):
        if verbose:
            print('Started load function')

        if not self.is_possible_batch(batch_size):
            raise ValueError('Too big batch size')

        data = []
        games_left = batch_size
        games_to_skip_left = self._from
        for player in os.walk(self._path):
            for game in player[2]:
                if games_to_skip_left > 0:
                    games_to_skip_left -= 1
                    continue

                if games_left > 0:
                    games_left -= 1
                else:
                    if verbose:
                        stdout.write('\rDone\n')
                    return data

                if verbose:
                    stdout.write('\rReading file: {}; {} files left'
                                 .format(os.path.join(player[0], game), games_left))

                game_data = []
                with open(os.path.join(player[0], game)) as text:
                    for line in text.readlines():
                        line = line[line.find('{'):].strip()
                        if line != '':
                            game_data.append(json.loads(line))
                data.append(game_data)
        return data

    def into_input(self, batch_size, verbose=1):
        if verbose:
            print('Started into_input function')

        data = self.load(batch_size=batch_size, verbose=verbose)
        x = []
        y = []
        for i, game_data in enumerate(data):
            if verbose:
                stdout.write('\rGames to parse left: {}'.format(batch_size - i))

            if len(game_data) != self._DATASET_NORMAL_SIZE:
                continue

            game_x = []
            game_y = []
            for step in game_data:
                if bool(step['finished']):
                    break

                turns_left = step['maxTurns'] - step['turn']
                heroes = []

                for hero in step['heroes']:
                    last_dir = self._POSSIBLE_DIRS.get(hero.get('lastDir'))
                    heroes.extend((int(hero['elo']), int(hero['pos']['x']), int(hero['pos']['y']),
                                   last_dir if last_dir is not None else 0, int(hero['life']), int(hero['gold']),
                                   int(hero['mineCount']), int(hero['spawnPos']['x']), int(hero['spawnPos']['y'])))

                board = [-1] * (28 ** 2 - step['board']['size'] ** 2)

                for index in range(2, step['board']['size'] ** 2 * 2, 2):
                    tile = step['board']['tiles'][index - 2] + step['board']['tiles'][index - 1]
                    board.append(self._POSSIBLE_TILES[tile])

                game_x.append(heroes + [turns_left] + board)

                expected_output = [0] * 5
                expected_output[heroes[3]] = 1
                game_y.append(expected_output)

            x.append(game_x)
            y.append(game_y)

        self._from += batch_size
        if verbose:
            stdout.write('\rDone\n')

        return array(x), array(y)

    def reset(self):
        self._from = 0

    def is_possible_batch(self, batch_size):
        return self._games_count - self._from >= batch_size
