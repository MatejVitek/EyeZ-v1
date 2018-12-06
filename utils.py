import os
import platform
import random
import re


def multi_replace(string, replacements, ignore_case=False):
	if ignore_case:
		replacements = dict((k.lower(), v) for (k, v) in replacements.items())
	rep = map(re.escape, sorted(replacements, key=len, reverse=True))
	pattern = re.compile("|".join(rep), re.I if ignore_case else 0)
	return pattern.sub(lambda match: replacements[match.group(0)], string)


class Info(object):
	def __init__(self, gender, age, color):
		self.gender = gender
		self.age = int(age)
		self.color = color


def get_disk_path(ext_disk_name=None, win_internal_drive='C:', win_external_drive='D:'):
	windows = platform.system().lower() == 'windows'
	if not ext_disk_name:
		return win_internal_drive if windows else '/hdd'
	return win_external_drive if windows else os.path.join('/media', os.getlogin(), ext_disk_name)


def get_rot_dir(*args, **kwargs):
	return os.path.join(get_disk_path(*args, **kwargs), 'EyeZ', 'Rot')


def get_id_info(path=os.path.join(get_rot_dir(), 'SBVP_Gender_Age_Color.txt')):
	with open(path, 'r') as f:
		return {
			(id + 1 + mirror_offset): Info(*line.split())
			for (id, line) in enumerate(f.readlines())
			for mirror_offset in (0, 55)
		}


def shuffle(l):
	random.shuffle(l)
	return l
