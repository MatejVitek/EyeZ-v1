import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np


class Figure(object):
	def __init__(self, name, colors, save, **kw):
		"""
		Wrapper for a figure and its corresponding settings

		:param name: Figure name
		:type  name: str or int or None
		:param colors: Color cycler
		:param save: File name to save figure to at the end
		:type  save: str or None
		:param str font_family: Name of font family to use for labels
		:param int font_size: Size of label fonts
		:param str xlabel: X label
		:param str ylabel: Y label
		:param x_tick_formatter: Tick formatter for x axis
		:param y_tick_formatter: Tick formatter for y axis
		:param tick_formatter: Same tick formatter for both axes
		:param xlim: Axis limits for x axis
		:param ylim: Axis limits for y axis
		:param lim: Same axis limits for both axes
		:param xticks: Tick positions and labels for x axis
		:param yticks: Tick positions and labels for y axis
		:param ticks: Same tick positions and labels for both axes
		:param bool grid: Should grid be drawn
		:param str grid_which: Which ticks to draw gridlines at
		:param str grid_axis: Which axes should have gridlines
		:param str grid_kw: Other keyword arguments to pass to :py:matplotlib.pyplot.grid
		:param legend_loc: Location of legend
		:param legend_size: Font size for legend
		:param margins: Margins for both axes
		:param pad: Padding for both axes
		"""

		self.name = name
		self.colors = colors
		self.color = next(self.colors)
		self.save = save
		self.settings = kw

	def __del__(self):
		if self.save:
			plt.figure(self.name)
			plt.savefig(self.save, bbox_inches='tight')

	def plot(self, x, y, **kw):
		cfg = self.settings

		plt.figure(self.name)
		plt.plot(x, y, color=self.color, linewidth=(kw.get('linewidth') or cfg.get('linewidth', 2)), **kw)

		plt.rcParams['font.family'] = cfg.get('font') or cfg.get('font_family', 'Times New Roman')
		plt.rcParams['font.size'] = cfg.get('font_size', 30)

		plt.xlabel(cfg.get('xlabel', ''))
		plt.xlabel(cfg.get('ylabel', ''))
		plt.gca().xaxis.set_major_formatter(
			cfg.get('x_tick_formatter') or
			cfg.get('tick_formatter', tick.FuncFormatter(def_tick_format))
		)
		plt.gca().yaxis.set_major_formatter(
			cfg.get('y_tick_formatter') or
			cfg.get('tick_formatter', tick.FuncFormatter(def_tick_format))
		)
		plt.xlim(*(cfg.get('xlim') or cfg.get('lim')))
		plt.ylim(*(cfg.get('ylim') or cfg.get('lim')))
		plt.xticks(cfg.get('xticks') or cfg.get('ticks'))
		plt.yticks(cfg.get('yticks') or cfg.get('ticks'))
		plt.grid(
			cfg.get('grid', True),
			which=cfg.get('grid_which', 'major'),
			axis=cfg.get('grid_axis', 'both'),
			**cfg.get('grid_kw', {})
		)
		plt.legend(loc=cfg.get('legend_loc', 'lower left'), fontsize=cfg.get('legend_size', 20))

		plt.margins(cfg.get('margins', 0))
		plt.tight_layout(pad=cfg.get('pad', 0))

	def next(self):
		self.color = next(self.colors)


class Painter(object):
	def __init__(self, figure=None, colors='plasma', k=None, interactive=True, save=None, **kw):
		"""
		Class for drawing multiple plots to a figure (or multiple figures)

		:param figure: Figure name or number
		:type  figure: str or int or None
		:param colors: Name of a colormap or a sequence of colors
		:param k: Number of colors to sample from the colormap. If None, the entire colormap will be used.
		:type  k: int or None
		:param bool interactive: Should execution continue after drawing
		:param kw: Default settings for all figures - see docs for :py:Figure
		"""

		self.figures = {}
		self.add_figure(figure, colors, k, save, **kw)
		self.default_settings = kw
		self.interactive = interactive
		if self.interactive:
			plt.ion()

	def __del__(self):
		if self.interactive:
			plt.ioff()
			plt.show()

	def add_figure(self, figure=None, colors='plasma', k=None, save=True, **kw):
		colors = cycle(colors, k)
		kw.update((k, v) for (k, v) in self.default_settings.items() if k not in kw)
		self.figures[figure] = Figure(figure, colors, save, **kw)

	def draw(self, x, y, figure=None, **kw):
		self.figures[figure].plot(x, y, **kw)
		if self.interactive:
			plt.draw()
			plt.pause(1)
		else:
			plt.show()


def cycle(colors, k=None):
	if isinstance(colors, str):
		colors = cm.get_cmap(colors)
	if k is None:
		while True:
			yield from colors
	else:
		idx = np.linspace(0, len(colors), k, endpoint=False, dtype=int)
		while True:
			for i in idx:
				yield colors[i]


def def_tick_format(x, _):
	return np.format_float_positional(x, precision=3, trim='-')
