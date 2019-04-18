import os
import re

import dataset
import utils


L = (dataset.L, 'left', 'Left', 'LEFT', 'l', 'L')
R = (dataset.R, 'right', 'Right', 'RIGHT', 'r', 'R')
C = (dataset.C, 'center', 'Center', 'CENTER', 'c', 'C', 's', 'S')
U = (dataset.U, 'up', 'Up', 'UP', 'u', 'U')


class NamingParser(object):
	def __init__(self, naming=r'ie_d_n', naming_re=None, eyes=r'LR', directions=r'lrsu', **kw):
		self.eyes = eyes
		if isinstance(self.eyes, dict):
			self.eyes = ''.join(self.eyes[d] for d in L + R if d in self.eyes)
		else:
			self.eyes = ''.join(self.eyes)

		self.directions = directions
		if isinstance(self.directions, dict):
			self.directions = ''.join(self.directions[d] for d in L + R + C + U if d in self.directions)
		else:
			self.directions = ''.join(self.directions)

		# Default taken from Iterator.white_list_formats in https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/iterator.py
		self.extensions = kw.get('valid_extensions', ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'))

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

		self.match = re.fullmatch if kw.get('strict', False) else re.match

	def parse(self, name):
		match = self.match(self.re, name)
		return {
			'id': int(match.group('id')),
			'eye': self.eyes.index(match.group('eye')),
			'direction': self.directions.index(match.group('direction')),
			'n': int(match.group('n'))
		}

	def valid(self, name_with_ext):
		name, ext = os.path.splitext(name_with_ext)
		return self.match(self.re, name) and any(ext.lower() in (x.lower(), '.' + x.lower()) for x in self.extensions)
