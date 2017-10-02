import numpy as np
import numpy.random as rng
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


def plot_logistic_2d_interactive():
	"""Plots a 2d logistic function. Has sliders to control parameters."""

	v = np.array([1, 1])
	b = 0

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

	scale_slider = Slider(plt.axes([0.05, 0.01, 0.2, 0.03]), '', 0, 8, valinit=1)
	scale_slider.on_changed(update)

	angle_slider = Slider(plt.axes([0.35, 0.01, 0.2, 0.03]), '', -np.pi, np.pi, valinit=0)
	angle_slider.on_changed(update)

	shift_slider = Slider(plt.axes([0.65, 0.01, 0.2, 0.03]), '', -10, 10, valinit=0)
	shift_slider.on_changed(update)

	plt.show()



def create_dataset(separable, seed=42):
	"""Creates a toy 2d dataset for logistic regression, either separable or not."""

	# clamp the random state
	state = rng.get_state()
	rng.seed(seed)

	# create data
	xs_pos = rng.randn(10, 2)
	xs_neg = rng.randn(10, 2)
	if separable:
		xs_pos[:, 0] += 3
		xs_neg[:, 0] -= 3
		xs_pos[:, 1] *= 3
		xs_neg[:, 1] *= 3
	else:
		xs_pos[:, 0] += 1
		xs_neg[:, 0] -= 1
	
	# restore the random state
	rng.set_state(state)

	return xs_pos, xs_neg


def manual_logistic_regression_demo(separable):
	"""Logistic regression interactive demo, where the user fits the weights manually."""

	v = np.array([1, 1])
	b = 0

	xlim = [-8, 8]
	xx = np.linspace(xlim[0], xlim[1], 50)
	X, Y = np.meshgrid(xx, xx)
	XY = np.dstack([X, Y])
	A = np.sum(XY * v, axis=2) + b
	Z = logistic(A)
	zlim = [0, 1]

	xs_pos, xs_neg = create_dataset(separable)

	fig = plt.figure()
	ax1 = plt.subplot(121, projection='3d')
	ax2 = plt.subplot(122)

	def calc_loss(v, b):
		losses_pos = np.log1p(np.exp(-np.dot(xs_pos, v) - b))
		losses_neg = np.log1p(np.exp(np.dot(xs_neg, v) + b))
		return np.mean(np.concatenate([losses_pos, losses_neg]))

	def plot_ax1(Z):
		ax1.cla()
		ax1.plot(xs_pos[:, 0], xs_pos[:, 1], 'b.', ms=12)
		ax1.plot(xs_neg[:, 0], xs_neg[:, 1], 'r.', ms=12)
		ax1.plot_wireframe(X, Y, Z)
		ax1.set_xlim(xlim)
		ax1.set_ylim(xlim)
		ax1.set_zlim(zlim)

	def plot_ax2(Z, v, b):
		ax2.cla()
		ax2.plot(xs_pos[:, 0], xs_pos[:, 1], 'b.', ms=12)
		ax2.plot(xs_neg[:, 0], xs_neg[:, 1], 'r.', ms=12)
		ax2.grid('on')
		ax2.axis('equal')
		CS = ax2.contour(X, Y, Z, colors='k', levels=[0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
		plt.setp(CS.collections[3], linewidth=4)
		origin = -b / np.dot(v, v) * v 
		ax2.arrow(origin[0], origin[1], v[0], v[1], lw=4, head_width=0.2, head_length=0.15, fc='g', ec='g')

	plot_ax1(Z)
	plot_ax2(Z, v, b)

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
	    loss_slider.set_val(np.log(calc_loss(new_v, new_b)))
	    fig.canvas.draw_idle()

	scale_slider = Slider(plt.axes([0.05, 0.03, 0.2, 0.02]), '', 0, 8, valinit=1)
	scale_slider.on_changed(update)

	angle_slider = Slider(plt.axes([0.35, 0.03, 0.2, 0.02]), '', -np.pi, np.pi, valinit=0)
	angle_slider.on_changed(update)

	shift_slider = Slider(plt.axes([0.65, 0.03, 0.2, 0.02]), '', -10, 10, valinit=0)
	shift_slider.on_changed(update)

	loss_slider = Slider(plt.axes([0.05, 0.93, 0.85, 0.02]), '', -10, 3, valinit=np.log(calc_loss(v, b)), color='red', dragging=False)

	plt.show()



def plot_logistic_regression_loss(separable):
	"""Plots the loss surface for logistic regression."""

	# create data
	xs_pos, xs_neg = create_dataset(separable)
	
	# a range of weights
	wrange = np.linspace(-20, 20, 50)
	W0, W1 = np.meshgrid(np.linspace(-15, 50, 50), np.linspace(-30, 30, 50))
	ws = np.stack([W0.flatten(), W1.flatten()])

	# the loss
	losses_pos = np.log1p(np.exp(-np.dot(xs_pos, ws)))
	losses_neg = np.log1p(np.exp(np.dot(xs_neg, ws)))
	total_loss = np.mean(np.concatenate([losses_pos, losses_neg], axis=0), axis=0)
	total_loss = np.reshape(total_loss, W0.shape)

	fontsize = 18
	markersize = 12

	# plot data
	fig = plt.figure()
	ax = plt.subplot(121)
	ax.plot(xs_pos[:, 0], xs_pos[:, 1], 'b.', ms=markersize)
	ax.plot(xs_neg[:, 0], xs_neg[:, 1], 'r.', ms=markersize)
	ax.axis('equal')
	ax.grid('on')
	ax.set_title('data', fontsize=fontsize)
	ax.set_xlabel(r'$x_1$', fontsize=fontsize)
	ax.set_ylabel(r'$x_2$', fontsize=fontsize)

	# plot loss
	ax = plt.subplot(122, projection='3d')
	ax.plot_wireframe(W0, W1, total_loss)
	ax.set_title('loss', fontsize=fontsize)
	ax.set_xlabel(r'$w_1$', fontsize=fontsize)
	ax.set_ylabel(r'$w_2$', fontsize=fontsize)
	
	plt.show()
	