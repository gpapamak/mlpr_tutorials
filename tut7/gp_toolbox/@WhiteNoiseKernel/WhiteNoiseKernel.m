classdef WhiteNoiseKernel < Kernel
% White noise kernel: produces gaussian white noise.
%
% George Papamakarios, Nov 2015

    properties (SetAccess = private, GetAccess = public)
        
        tol = 1.0e-9;
        
    end

    methods (Access = protected)
               
        % evaluate kernel: one data matrix
        function [S] = eval_x(obj, x)
            
            % number of datapoints
            Nx = size(x, 2);
            
            % evaluate kernel
            A1 = x' * x;
            A2 = diag(A1) * ones(1, Nx) / 2;
            S = double(abs(A1 - A2 - A2') < obj.tol);
        end 
        
        % evaluate kernel: two data matrices
        function [S] = eval_xy(obj, x, y)
            
            % check dimensionality
            assert(size(x, 1) == size(y, 1), 'Dimensionalities don''t match.');
            
            % number of datapoints
            Nx = size(x, 2);
            Ny = size(y, 2);
            
            % evaluate kernel
            Ax = sum(x.^2, 1)' * ones(1, Ny) / 2;
            Ay = ones(Nx, 1) * sum(y.^2, 1) / 2;
            Axy = x' * y;
            S = double(abs(Axy - Ax - Ay) < obj.tol);
        end
        
        % evaluate kernel and derivatives: one data matrix
        function [S] = eval_x_d(~, ~) %#ok<STOUT>
            error('GaussianProcesses:nonDifferentiable', 'White noise is not differentiable.');
        end 
        
        % evaluate kernel and derivatives: two data matrices
        function [S] = eval_xy_d(~, ~, ~) %#ok<STOUT>
            error('GaussianProcesses:nonDifferentiable', 'White noise is not differentiable.');
        end
        
    end
end
