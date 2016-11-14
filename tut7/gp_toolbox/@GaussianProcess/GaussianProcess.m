classdef GaussianProcess < handle
% Implements a gaussian process.
%
% George Papamakarios, Nov 2015

    properties (SetAccess = private, GetAccess = public)
        
        kf = []
        
    end

    methods (Access = public)
        
        % constructor
        function [obj] = GaussianProcess(kernel)
            
            % check inputs
            if ~isa(kernel, 'Kernel')
                kernel = ConstantKernel(kernel);
            end
            assert(isa(kernel, 'Kernel'), 'Invalid kernel object.');
                        
            % set parameters
            obj.kf = kernel;
        end 
        
        % evaluates quadratic and its derivative for a given input
        function [y, dydx] = eval(obj, x)
            
            [Dx, Nx] = size(x);
                        
            if nargout < 2
                m = zeros(1, Nx, 'like', x);
                y = gauss_sample(m, obj.kf.eval(x));
            else
                m = zeros(1, Nx*(1+Dx), 'like', x);
                y_dydx = gauss_sample(m, obj.kf.eval_d(x));
                y = y_dydx(1:Nx)';
                dydx = reshape(y_dydx(Nx+1:end), Dx, Nx);
                dydx = permute(dydx, [1 3 2]);
            end
        end
        
    end
end
