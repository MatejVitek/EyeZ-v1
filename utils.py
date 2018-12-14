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


class Bin(object):
	def __init__(self, label, samples):
		self.label = label
		self.samples = samples


def get_bins(samples, by='age', bins=None, get_id='id', bin_labels=None):
	id_info = get_id_info()

	if not get_id:
		ids = map(int, samples)
	elif isinstance(get_id, str):
		ids = [int(getattr(s, get_id)) for s in samples]
	else:
		ids = [int(get_id(s)) for s in samples]

	f = (lambda id: getattr(id_info[id], by)) if isinstance(by, str) else by
	bins = sorted(bins or set(f(id) for id in ids))

	try:
		# Check if all bins are numbers
		all(bin < float('inf') for bin in bins)
		return [
			# If bin labels not given, use "left_bin_border <= by < right_bin_border"
			Bin(
				bin_labels[i] if bin_labels else (
					((str(bins[i - 1]) + " > ") if i > 0 else "")
					+ f'{by}'
					+ ((" <= " + str(bins[i])) if i < len(bins) else "")
				),
				[
					s
					for (s, id) in zip(samples, ids)
					if (i == 0 or f(id) > bins[i-1])
					and (i == len(bins) or f(id) <= bins[i])
				]
			) for i in range(len(bins) + 1)
		]
	except TypeError:
		# If bin labels not given, use bin names
		return [
			Bin(bin_label, [s for (s, id) in zip(samples, ids) if f(id) == bin])
			for bin_label, bin in zip((bin_labels or bins), bins)
		]
