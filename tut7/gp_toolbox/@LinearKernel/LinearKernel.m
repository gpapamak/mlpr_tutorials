classdef LinearKernel < Kernel
% Linear kernel: k(x,y) = sum( si^2 * xi * yi ). Produces linear functions.
%
% George Papamakarios, Nov 2015

    properties (SetAccess = private, GetAccess = public)
        
        s = []
        
    end
    
    methods (Access = protected)
               
        % evaluate kernel: one data matrix
        function [S] = eval_x(obj, x)
            
            % scale inputs
            if isscalar(obj.s)
                x = x * obj.s;
            else
                Nx = size(x, 2);
                x = x .* (obj.s * ones(1, Nx));
            end
            
            % evaluate kernel
            S = x' * x;
                        
            % for numerical stability
            S = (S + S') / 2;
        end 
        
        % evaluate kernel: two data matrices
        function [S] = eval_xy(obj, x, y)
            
            % check dimensionality
            assert(size(x, 1) == size(y, 1), 'Dimensionalities don''t match.');
            
            % number of datapoints
            Nx = size(x, 2);
            Ny = size(y, 2);
            
            % scale inputs
            if isscalar(obj.s)
                x = x * obj.s;
                y = y * obj.s;
            else
                x = x .* (obj.s * ones(1, Nx));
                y = y .* (obj.s * ones(1, Ny));
            end
            
            % evaluate kernel
            S = x' * y;
        end
        
        % evaluate kernel and derivatives: one data matrix
        function [S] = eval_x_d(obj, x)
            
            [Dx, Nx] = size(x);
            Sxx = eval_x(obj, x);
            
            if isscalar(obj.s)
                Sdxx = x * obj.s^2;
                Sdxdx = obj.s^2 * eye(Dx);
            else
                Sdxx = x .* (obj.s.^2 * ones(1, Nx));
                Sdxdx = diag(obj.s.^2);
            end
            
            Sdxx = repmat(Sdxx, Nx, 1);
            Sdxdx = repmat(Sdxdx, Nx, Nx);
            
            S = zeros(Nx*(Dx + 1), 'like', x);
            S(1:Nx, 1:Nx) = Sxx;
            S(Nx+1:end, 1:Nx) = Sdxx;
            S(1:Nx, Nx+1:end) = Sdxx';
            S(Nx+1:end, Nx+1:end) = Sdxdx;
        end 
        
        % evaluate kernel and derivatives: two data matrices
        function [S] = eval_xy_d(obj, x, y)
            
            [Dx, Nx] = size(x);
            [Dy, Ny] = size(y);
            assert(Dx == Dy, 'Dimensionalities dont''t match.');
            Sxy = eval_xy(obj, x, y);
            
            if isscalar(obj.s)
                Sdxy = y * obj.s^2;
                Sxdy = x * obj.s^2;
                Sdxdy = obj.s^2 * eye(Dx,Dy);
            else
                Sdxy = y .* (obj.s.^2 * ones(1, Ny));
                Sxdy = x .* (obj.s.^2 * ones(1, Nx));
                Sdxdy = diag(obj.s.^2);
            end
            
            Sdxy = repmat(Sdxy, Nx, 1);
            Sxdy = repmat(Sxdy, Ny, 1);
            Sdxdy = repmat(Sdxdy, Nx, Ny);
            
            S = zeros(Nx*(Dx + 1), Ny*(Dy + 1), 'like', x);
            S(1:Nx, 1:Ny) = Sxy;
            S(Nx+1:end, 1:Ny) = Sdxy;
            S(1:Nx, Ny+1:end) = Sxdy';
            S(Nx+1:end, Ny+1:end) = Sdxdy;
        end
        
    end

    methods (Access = public)
        
        % constructor
        function [obj] = LinearKernel(s)
            
            s = s(:);
            assert(all(s) > 0, 'Scale parameters must be positive.');
            obj.s = s;
        end
    end
end
