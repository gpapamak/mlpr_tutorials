% visualizes samples from a gaussian process

clear;
addpath(fullfile(pwd, 'gp_toolbox'));

% construct the kernel
kernel = 0.01 * SquareExponentialKernel(1) + 1;

% construct the gaussian process
gp = GaussianProcess(kernel);

% number of samples to show
n_samples = 10;

% draw and plot samples
figure; hold on;
xx = linspace(-10, 10, 1000);
for i = 1:n_samples
    y = gp.eval(xx);
    plot(xx, y);
end
