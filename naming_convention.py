import re

import cross_validate
import utils


L = (cross_validate.L, 'left', 'Left', 'LEFT', 'l', 'L')
R = (cross_validate.R, 'right', 'Right', 'RIGHT', 'r', 'R')
C = (cross_validate.C, 'center', 'Center', 'CENTER', 'c', 'C')
U = (cross_validate.U, 'up', 'Up', 'UP', 'u', 'U')


class NamingHandler(object):
	def __init__(self, naming_re=None, naming=r'ie_d_n', eyes=r'LR', directions=r'lrsu', strict=False):
		self.eyes = eyes
		if isinstance(self.eyes, dict):
			self.eyes = ''.join(self.eyes[d] for d in L + R if d in self.eyes.keys())
		else:
			self.eyes = ''.join(self.eyes)

		self.directions = directions
		if isinstance(self.directions, dict):
			self.directions = ''.join(self.directions[d] for d in L + R + C + U if d in self.directions.keys())
		else:
			self.directions = ''.join(self.directions)

		if naming_re:
			self.re = naming_re
		else:
			self.re = naming
			rep = {
				'i': r'(?P<id>\d+)',
				'e': rf'(?P<eye>[{self.eyes}])',
				'd': rf'(?P<direction>[{self.directions}])',
				'n': r'(?P<n>\d+)'
			}
			self.re = utils.multi_replace(re.escape(self.re), rep, True)

		for pattern in (r'(?P<id>', r'(?P<eye>', r'(?P<direction>', r'(?P<n>'):
			if pattern not in self.re:
				raise ValueError(f"Missing pattern {pattern} in naming regex.")

		self.match = re.fullmatch if strict else re.match

	def parse(self, name):
		match = self.match(self.re, name)
		return {
			'id': int(match.group('id')),
			'eye': self.eyes.index(match.group('eye')),
			'direction': self.directions.index(match.group('direction')),
			'n': int(match.group('n'))
		}

	def valid(self, name):
		return self.match(self.re, name)
