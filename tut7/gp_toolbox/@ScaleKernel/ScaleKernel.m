classdef ScaleKernel < Kernel
% Composite kernel, given by a scaled kernel.
%
% George Papamakarios, Nov 2015

    properties (SetAccess = private, GetAccess = public)
        
        c = []
        k = []
        
    end
    
    methods (Access = protected)
               
        % evaluate kernel: one data matrix
        function [S] = eval_x(obj, x)
            
            S = obj.c * obj.k.eval_x(x);
        end 
        
        % evaluate kernel: two data matrices
        function [S] = eval_xy(obj, x, y)
            
            S = obj.c * obj.k.eval_xy(x, y);
        end
        
        % evaluate kernel and derivatives: one data matrix
        function [S] = eval_x_d(obj, x)
            
            S = obj.c * obj.k.eval_x_d(x);
        end 
        
        % evaluate kernel and derivatives: two data matrices
        function [S] = eval_xy_d(obj, x, y)
            
            S = obj.c * obj.k.eval_xy_d(x, y);
        end
        
    end

    methods (Access = public)
        
        % constructor
        function [obj] = ScaleKernel(c, k)
            
            % take care of constants
            if ~isa(k, 'Kernel')
                k = ConstantKernel(k);
            end
            
            % check inputs
            assert(isscalar(c) && c >= 0, 'Constant must be a non-negative scalar.');
            assert(isa(k, 'Kernel'), 'Input is not a valid kernel.');
                        
            % set parameters
            obj.c = c;
            obj.k = k;
        end
    end
end
