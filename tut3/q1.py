import numpy as np
import matplotlib.pyplot as plt


x1 = np.array([0.5, 0.1, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.35, 0.25])
x2 = np.array([0.9, 0.8, 0.75, 1.0])


lims = [-3.0, 4.0]


def log_gaussian(xs, m, v):

	return -0.5 * ((xs - m) ** 2 / v + np.log(2.0 * np.pi * v))


def fit_model():

	m1 = np.mean(x1)
	m2 = np.mean(x2)

	v1 = np.var(x1)
	v2 = np.var(x2)

	p1 = float(x1.size) / (x1.size + x2.size)
	p2 = 1.0 - p1

	return m1, m2, v1, v2, p1, p2


def print_fitted_model():

	m1, m2, v1, v2, p1, p2 = fit_model()

	print 'm1 = {0}'.format(m1)
	print 'm2 = {0}'.format(m2)
	print 'v1 = {0}'.format(v1)
	print 'v2 = {0}'.format(v2)
	print 'p1 = {0}'.format(p1)
	print 'p2 = {0}'.format(p2)


def calc_prob(x):

	m1, m2, v1, v2, p1, p2 = fit_model()

	s1 = log_gaussian(x, m1, v1) + np.log(p1)
	s2 = log_gaussian(x, m2, v2) + np.log(p2)
	d1 = 1. / (1. + np.exp(s2 - s1))
	
	print 'P(y=1 | x={0}) = {1}'.format(x, d1)


def plot_scores(log=False):

	m1, m2, v1, v2, p1, p2 = fit_model()

	xs = np.linspace(lims[0], lims[1], 1000)
	s1 = log_gaussian(xs, m1, v1) + np.log(p1)
	s2 = log_gaussian(xs, m2, v2) + np.log(p2)
	d1 = 1. / (1. + np.exp(s2 - s1))
	d2 = 1. - d1

	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	f = (lambda u: u) if log else np.exp
	ax1.plot(xs, f(s1), 'b', label='P(y=1) p(x|y=1)', linewidth=2)
	ax1.plot(xs, f(s2), 'r', label='P(y=1) p(x|y=2)', linewidth=2)
	ax2.plot(xs, d1, 'b', label='P(y=1|x)', linewidth=2)
	ax2.plot(xs, d2, 'r', label='P(y=2|x)', linewidth=2)
	ax1.set_xlim(lims)
	ax2.set_xlim(lims)
	ax2.set_ylim([0, 1.5])
	ax1.legend()
	ax2.legend()
	plt.show()
