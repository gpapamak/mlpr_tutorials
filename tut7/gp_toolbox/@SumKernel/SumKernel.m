classdef SumKernel < Kernel
% Composite kernel, given by the sum of two kernels.
%
% George Papamakarios, Nov 2015

    properties (SetAccess = private, GetAccess = public)
        
        k1 = []
        k2 = []
        
    end
    
    methods (Access = protected)
               
        % evaluate kernel: one data matrix
        function [S] = eval_x(obj, x)
            
            S = obj.k1.eval_x(x) + obj.k2.eval_x(x);
        end 
        
        % evaluate kernel: two data matrices
        function [S] = eval_xy(obj, x, y)
            
            S = obj.k1.eval_xy(x, y) + obj.k2.eval_xy(x, y);
        end
        
        % evaluate kernel and derivatives: one data matrix
        function [S] = eval_x_d(obj, x)
            
            S = obj.k1.eval_x_d(x) + obj.k2.eval_x_d(x);
        end 
        
        % evaluate kernel and derivatives: two data matrices
        function [S] = eval_xy_d(obj, x, y)
            
            S = obj.k1.eval_xy_d(x, y) + obj.k2.eval_xy_d(x, y);
        end
        
    end

    methods (Access = public)
        
        % constructor
        function [obj] = SumKernel(k1, k2)
            
            % take care of constants
            if ~isa(k1, 'Kernel')
                k1 = ConstantKernel(k1);
            end
            if ~isa(k2, 'Kernel')
                k2 = ConstantKernel(k2);
            end
            
            % check inputs
            assert(isa(k1, 'Kernel') && isa(k2, 'Kernel'), 'Input objects are not valid kernels.');
                        
            % set parameters
            obj.k1 = k1;
            obj.k2 = k2;
        end
    end
end
