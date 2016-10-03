import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def plot_rbfs(h):
	"""Plots the radial basis functions for a given width h."""

	K = 101
	cs = (np.arange(K) - K/2) * h / np.sqrt(2)
	xs = np.linspace(cs[0] - 3*h, cs[-1] + 3*h, 1000)[:, np.newaxis]
	ys = np.exp(-((xs - cs)/h) ** 2)

	fig, ax = plt.subplots(1, 1)
	ax.plot(xs, ys, color='gray')
	ax.plot(xs, ys[:, 0], lw=3)
	ax.plot(xs, ys[:, K/2], lw=3)
	ax.plot(xs, ys[:, -1], lw=3)

	ax.set_ylim([0, 1.4])
	ax.set_title('h = {0}'.format(h))
	plt.show()


def plot_rbfs_interactive():
	"""Plots the radial basis functions. Interactively control the width h with a slider."""

	h = 0.2
	K = 101
	xlim = [-8, 8]
	cs = (np.arange(K) - K/2) * h / np.sqrt(2)
	xs = np.linspace(xlim[0], xlim[1], (xlim[1] - xlim[0]) / 0.01)[:, np.newaxis]
	ys = np.exp(-((xs - cs)/h) ** 2)

	fig, ax = plt.subplots(1, 1)
	lines = ax.plot(xs, ys, color='gray')
	lineL, = ax.plot(xs, ys[:, 0], lw=3)
	lineM, = ax.plot(xs, ys[:, K/2], lw=3)
	lineR, = ax.plot(xs, ys[:, -1], lw=3)
	ax.set_xlim(xlim)
	ax.set_ylim([0, 1.4])

	def h_update(val):
	    h = h_slider.val
	    cs = (np.arange(K) - K/2) * h / np.sqrt(2)
	    ys = np.exp(-((xs - cs)/h) ** 2)
	    for i, line in enumerate(lines):
	    	line.set_ydata(ys[:, i])
	    lineL.set_ydata(ys[:, 0])
	    lineM.set_ydata(ys[:, K/2])
	    lineR.set_ydata(ys[:, -1])
	    fig.canvas.draw_idle()

	h_axis = plt.axes([0.3, 0.8, 0.4, 0.03])
	h_slider = Slider(h_axis, 'h', 0.01, 1.5, valinit=h)
	h_slider.on_changed(h_update)

	plt.show()
