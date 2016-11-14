classdef ConstantKernel < Kernel
% Constant kernel: just returns a constant covariance matrix. Mainly used
% to construct more complicated kernels.
%
% George Papamakarios, Nov 2015

    properties (SetAccess = private, GetAccess = public)
        
        c = [];
        
    end
    
    methods (Access = protected)
               
        % evaluate kernel: one data matrix
        function [S] = eval_x(obj, x)
            
            Nx = size(x, 2);
            S = obj.c * ones(Nx, Nx, 'like', x);
        end 
        
        % evaluate kernel: two data matrices
        function [S] = eval_xy(obj, x, y)
            
            assert(size(x, 1) == size(y, 1), 'Dimensionalities dont''t match.');
            Nx = size(x, 2);
            Ny = size(y, 2);
            S = obj.c * ones(Nx, Ny, 'like', x);
        end
        
        % evaluate kernel and derivatives: one data matrix
        function [S] = eval_x_d(obj, x)
            
            [Dx, Nx] = size(x);
            S = zeros(Nx*(Dx + 1), 'like', x);
            S(1:Nx, 1:Nx) = eval_x(obj, x);
        end 
        
        % evaluate kernel and derivatives: two data matrices
        function [S] = eval_xy_d(obj, x, y)
            
            [Dx, Nx] = size(x);
            [Dy, Ny] = size(y);
            assert(Dx == Dy, 'Dimensionalities dont''t match.');
            S = zeros(Nx*(Dx + 1), Ny*(Dy + 1), 'like', x);
            S(1:Nx, 1:Ny) = eval_xy(obj, x, y);
        end
        
    end

    methods (Access = public)
        
        % constructor
        function [obj] = ConstantKernel(c)

            assert(isscalar(c) && c >= 0, 'Constant must be a non-negative scalar.');
            obj.c = c;
        end
    end
end
