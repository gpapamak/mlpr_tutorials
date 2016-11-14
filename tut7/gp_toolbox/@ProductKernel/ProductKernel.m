classdef ProductKernel < Kernel
% Composite kernel, given by the product of two kernels.
%
% George Papamakarios, Nov 2015

    properties (SetAccess = private, GetAccess = public)
        
        k1 = []
        k2 = []
        
    end
    
    methods (Access = protected)
               
        % evaluate kernel: one data matrix
        function [S] = eval_x(obj, x)
            
            S = obj.k1.eval_x(x) .* obj.k2.eval_x(x);
        end 
        
        % evaluate kernel: two data matrices
        function [S] = eval_xy(obj, x, y)
            
            S = obj.k1.eval_xy(x, y) .* obj.k2.eval_xy(x, y);
        end
        
        % evaluate kernel and derivatives: one data matrix
        function [S] = eval_x_d(obj, x)
            
            [Dx, Nx] = size(x);
            
            S1 = obj.k1.eval_x_d(x);
            S2 = obj.k2.eval_x_d(x);
            
            xx = 1 : Nx;
            dx = Nx + 1 : Nx*(Dx + 1);
            
            S = zeros(Nx*(Dx + 1), 'like', x);
            S(xx, xx) = S1(xx, xx) .* S2(xx, xx);
            S(dx, xx) = kron(S1(xx, xx), ones(Dx, 1)) .* S2(dx, xx) + kron(S2(xx, xx), ones(Dx, 1)) .* S1(dx, xx);
            S(xx, dx) = kron(S1(xx, xx), ones(1, Dx)) .* S2(xx, dx) + kron(S2(xx, xx), ones(1, Dx)) .* S1(xx, dx);
            S(dx, dx) = S1(dx, dx) .* kron(S2(xx, xx), ones(Dx, Dx)) + S2(dx, dx) .* kron(S1(xx, xx), ones(Dx, Dx)) + ...
                kron(S1(dx, xx), ones(1, Dx)) .* kron(S2(xx, dx), ones(Dx, 1)) + kron(S2(dx, xx), ones(1, Dx)) .* kron(S1(xx, dx), ones(Dx, 1));
            
            S = (S + S') / 2;
        end 
        
        % evaluate kernel and derivatives: two data matrices
        function [S] = eval_xy_d(obj, x, y)
            
            [Dx, Nx] = size(x);
            [Dy, Ny] = size(y);
            
            S1 = obj.k1.eval_xy_d(x, y);
            S2 = obj.k2.eval_xy_d(x, y);
            
            xx = 1 : Nx;
            yy = 1 : Ny;
            dx = Nx + 1 : Nx*(Dx + 1);
            dy = Ny + 1 : Ny*(Dy + 1);
            
            S = zeros(Nx*(Dx + 1), Ny*(Dy + 1), 'like', x);
            S(xx, yy) = S1(xx, yy) .* S2(xx, yy);
            S(dx, yy) = kron(S1(xx, yy), ones(Dx, 1)) .* S2(dx, yy) + kron(S2(xx, yy), ones(Dx, 1)) .* S1(dx, yy);
            S(xx, dy) = kron(S1(xx, yy), ones(1, Dy)) .* S2(xx, dy) + kron(S2(xx, yy), ones(1, Dy)) .* S1(xx, dy);
            S(dx, dy) = S1(dx, dy) .* kron(S2(xx, yy), ones(Dx, Dy)) + S2(dx, dy) .* kron(S1(xx, yy), ones(Dx, Dy)) + ...
                kron(S1(dx, yy), ones(1, Dy)) .* kron(S2(xx, dy), ones(Dx, 1)) + kron(S2(dx, yy), ones(1, Dy)) .* kron(S1(xx, dy), ones(Dx, 1));
        end
        
    end

    methods (Access = public)
        
        % constructor
        function [obj] = ProductKernel(k1, k2)
            
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
