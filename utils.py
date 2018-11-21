import re


def multi_replace(string, replacements, ignore_case=False):
	if ignore_case:
		replacements = dict((k.lower(), v) for (k, v) in replacements.items())
	rep = map(re.escape, sorted(replacements, key=len, reverse=True))
	pattern = re.compile("|".join(rep), re.I if ignore_case else 0)
	return pattern.sub(lambda match: replacements[match.group(0)], string)

