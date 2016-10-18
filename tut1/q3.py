import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider


matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('text', usetex=True)


def logistic(x):
	"""Calculates the logistic function elementwise."""

	x = np.asarray(x)
	return 1 / (1 + np.exp(-x))


def plot_logistic_1d():
	"""Plots a 1d logistic."""

	xlim = [-8, 8]
	xs = np.linspace(xlim[0], xlim[1], 100)
	ys = logistic(xs)

	fig, ax = plt.subplots(1, 1)
	ax.plot(xs, ys)
	ax.grid('on')
	ax.set_xlim(xlim)
	ax.set_ylim([-0.5, 1.5])
	ax.set_title('y = 1 / (1 + exp(-x))')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	plt.show(block=False)


def plot_linear_2d():
	"""Plots a 2d linear function."""

	v = [1, 2]
	b = 5

	xlim = [-8, 8]
	xx = np.linspace(xlim[0], xlim[1], 50)
	X, Y = np.meshgrid(xx, xx)
	XY = np.dstack([X, Y])
	A = np.sum(XY * v, axis=2) + b

	fig = plt.figure()
	ax = plt.subplot(121, projection='3d')
	ax.plot_wireframe(X, Y, A)

	ax = plt.subplot(122)
	ax.grid('on')
	ax.axis('equal')
	CS = ax.contour(X, Y, A, colors='k', levels=[-24, -16, -8, 0, 8, 16, 24])
	plt.setp(CS.collections[3], linewidth=4)
	ax.arrow(0, 0, v[0], v[1], lw=4, head_width=0.2, head_length=0.15, fc='r', ec='r')

	fig.suptitle(r'$y = v^Tx+b$', fontsize=20)
	plt.show(block=False)


def plot_logistic_2d():
	"""Plots a 2d logistic function."""

	v = [1, 2]
	b = 5

	xlim = [-8, 8]
	xx = np.linspace(xlim[0], xlim[1], 50)
	X, Y = np.meshgrid(xx, xx)
	XY = np.dstack([X, Y])
	A = np.sum(XY * v, axis=2) + b
	Z = logistic(A)

	fig = plt.figure()
	ax = plt.subplot(121, projection='3d')
	ax.plot_wireframe(X, Y, Z)

	ax = plt.subplot(122)
	ax.grid('on')
	ax.axis('equal')
	CS = ax.contour(X, Y, Z, colors='k', levels=[0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]);
	plt.setp(CS.collections[3], linewidth=4)
	ax.arrow(0, 0, v[0], v[1], lw=4, head_width=0.2, head_length=0.15, fc='r', ec='r')

	fig.suptitle(r'$y = \sigma(v^Tx+b)$', fontsize=20)
	plt.show(block=False)


def plot_linear_2d_interactive():
	"""Plots a 2d linear function. Has sliders to control parameters."""

	v = np.array([1, 2])
	b = 5

	xlim = [-8, 8]
	xx = np.linspace(xlim[0], xlim[1], 50)
	X, Y = np.meshgrid(xx, xx)
	XY = np.dstack([X, Y])
	A = np.sum(XY * v, axis=2) + b
	zlim = [np.min(A), np.max(A)]

	fig = plt.figure()
	ax1 = plt.subplot(121, projection='3d')
	ax2 = plt.subplot(122)

	def plot_ax1(A):
		ax1.cla()
		ax1.plot_wireframe(X, Y, A)
		ax1.set_xlim(xlim)
		ax1.set_ylim(xlim)
		ax1.set_zlim(zlim)

	def plot_ax2(A, v, b):
		ax2.cla()
		ax2.grid('on')
		ax2.axis('equal')
		CS = ax2.contour(X, Y, A, colors='k', levels=[-24, -16, -8, 0, 8, 16, 24])
		plt.setp(CS.collections[3], linewidth=4)
		origin = -b / np.dot(v, v) * v
		ax2.arrow(origin[0], origin[1], v[0], v[1], lw=4, head_width=0.2, head_length=0.15, fc='g', ec='g')

	plot_ax1(A)
	plot_ax2(A, v, b)
	fig.suptitle(r'$y = v^Tx+b$', fontsize=20)

	def update(val):
	    scale = scale_slider.val
	    angle = angle_slider.val
	    shift = shift_slider.val
	    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
	    new_v = scale * np.dot(rot, v)
	    new_b = scale * (b + shift)
	    new_A =  np.sum(XY * new_v, axis=2) + new_b
	    plot_ax1(new_A)
	    plot_ax2(new_A, new_v, new_b)
	    fig.canvas.draw_idle()

	scale_slider = Slider(plt.axes([0.05, 0.01, 0.2, 0.03]), '', 0, 2, valinit=1)
	scale_slider.on_changed(update)

	angle_slider = Slider(plt.axes([0.35, 0.01, 0.2, 0.03]), '', -np.pi, np.pi, valinit=0)
	angle_slider.on_changed(update)

	shift_slider = Slider(plt.axes([0.65, 0.01, 0.2, 0.03]), '', -5, 5, valinit=0)
	shift_slider.on_changed(update)

	plt.show()


def plot_logistic_2d_interactive():
	"""Plots a 2d logistic function. Has sliders to control parameters."""

	v = np.array([1, 2])
	b = 5

	xlim = [-8, 8]
	xx = np.linspace(xlim[0], xlim[1], 50)
	X, Y = np.meshgrid(xx, xx)
	XY = np.dstack([X, Y])
	A = np.sum(XY * v, axis=2) + b
	Z = logistic(A)
	zlim = [0, 1]

	fig = plt.figure()
	ax1 = plt.subplot(121, projection='3d')
	ax2 = plt.subplot(122)

	def plot_ax1(Z):
		ax1.cla()
		ax1.plot_wireframe(X, Y, Z)
		ax1.set_xlim(xlim)
		ax1.set_ylim(xlim)
		ax1.set_zlim(zlim)

	def plot_ax2(Z, v, b):
		ax2.cla()
		ax2.grid('on')
		ax2.axis('equal')
		CS = ax2.contour(X, Y, Z, colors='k', levels=[0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
		plt.setp(CS.collections[3], linewidth=4)
		origin = -b / np.dot(v, v) * v
		ax2.arrow(origin[0], origin[1], v[0], v[1], lw=4, head_width=0.2, head_length=0.15, fc='g', ec='g')


	plot_ax1(Z)
	plot_ax2(Z, v, b)
	fig.suptitle(r'$y = \sigma(v^Tx+b)$', fontsize=20)

	def update(val):
	    scale = scale_slider.val
	    angle = angle_slider.val
	    shift = shift_slider.val
	    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
	    new_v = scale * np.dot(rot, v)
	    new_b = scale * (b + shift)
	    new_A = np.sum(XY * new_v, axis=2) + new_b
	    new_Z = logistic(new_A)
	    plot_ax1(new_Z)
	    plot_ax2(new_Z, new_v, new_b)
	    fig.canvas.draw_idle()

	scale_slider = Slider(plt.axes([0.05, 0.01, 0.2, 0.03]), '', 0, 2, valinit=1)
	scale_slider.on_changed(update)

	angle_slider = Slider(plt.axes([0.35, 0.01, 0.2, 0.03]), '', -np.pi, np.pi, valinit=0)
	angle_slider.on_changed(update)

	shift_slider = Slider(plt.axes([0.65, 0.01, 0.2, 0.03]), '', -5, 5, valinit=0)
	shift_slider.on_changed(update)

	plt.show()
