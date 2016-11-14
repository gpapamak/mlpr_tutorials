classdef SquareExponentialKernel < Kernel
% Square exponential kernel: k(x,y) = exp(-1/2 sum( (xi - yi)^2 / li^2 )).
%
% George Papamakarios, Nov 2015

    properties (SetAccess = private, GetAccess = public)
        
        l = []
        
    end
    
    methods (Access = protected)
               
        % evaluate kernel: one data matrix
        function [S] = eval_x(obj, x)
            
            % number of datapoints
            Nx = size(x, 2);
            
            % scale inputs
            if isscalar(obj.l)
                x = x / obj.l;
            else
                x = x ./ (obj.l * ones(1, Nx));
            end
            
            % evaluate kernel
            A1 = x' * x;
            A2 = diag(A1) * ones(1, Nx) / 2;
            S = exp(A1 - A2 - A2');
            
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
            if isscalar(obj.l)
                x = x / obj.l;
                y = y / obj.l;
            else
                x = x ./ (obj.l * ones(1, Nx));
                y = y ./ (obj.l * ones(1, Ny));
            end
            
            % evaluate kernel
            Ax = sum(x.^2, 1)' * ones(1, Ny) / 2;
            Ay = ones(Nx, 1) * sum(y.^2, 1) / 2;
            Axy = x' * y;
            S = exp(Axy - Ax - Ay);
        end
        
        % evaluate kernel and derivatives: one data matrix
        function [S] = eval_x_d(obj, x)
            
            [Dx, Nx] = size(x);
            kxx = eval_x(obj, x);
            
            if isscalar(obj.l)
                x = x / obj.l^2;
                diag_Sdxdx = eye(Dx) / obj.l^2;
            else
                x = x ./ (obj.l.^2 * ones(1, Nx));
                diag_Sdxdx = diag(1./obj.l.^2);
            end
            
            xmx = x(:) * ones(1, Nx) - repmat(x, Nx, 1);
            Sdxx = -xmx .* kron(kxx, ones(Dx, 1));
            
            S = zeros(Nx*(Dx + 1), 'like', x);
            S(1:Nx, 1:Nx) = kxx;
            S(Nx+1:end, 1:Nx) = Sdxx;
            S(1:Nx, Nx+1:end) = Sdxx';
            S(Nx+1:end, Nx+1:end) = (repmat(diag_Sdxdx, Nx, Nx) + kron(xmx, ones(1, Dx)) .* kron(xmx', ones(Dx, 1))) .* kron(kxx, ones(Dx, Dx));
            
            S = (S + S') / 2;
        end
        
        % evaluate kernel and derivatives: two data matrices
        function [S] = eval_xy_d(obj, x, y)
            
            [Dx, Nx] = size(x);
            [Dy, Ny] = size(y);
            assert(Dx == Dy, 'Dimensionalities don''t match.');
            kxy = eval_xy(obj, x, y);
            
            if isscalar(obj.l)
                x = x / obj.l^2;
                y = y / obj.l^2;
                diag_Sdxdy = eye(Dx, Dy) / obj.l^2;
            else
                x = x ./ (obj.l.^2 * ones(1, Nx));
                y = y ./ (obj.l.^2 * ones(1, Ny));
                diag_Sdxdy = diag(1./obj.l.^2);
            end
            
            xmyv = x(:) * ones(1, Ny) - repmat(y, Nx, 1);
            Sdxy = -xmyv .* kron(kxy, ones(Dx, 1));
            
            xmyh = repmat(x', 1, Ny) - ones(Nx, 1) * y(:)';
            Sxdy = xmyh .* kron(kxy, ones(1, Dy));
            
            S = zeros(Nx*(Dx + 1), Ny*(Dy + 1), 'like', x);
            S(1:Nx, 1:Ny) = kxy;
            S(Nx+1:end, 1:Ny) = Sdxy;
            S(1:Nx, Ny+1:end) = Sxdy;
            S(Nx+1:end, Ny+1:end) = (repmat(diag_Sdxdy, Nx, Ny) - kron(xmyv, ones(1, Dy)) .* kron(xmyh, ones(Dx, 1))) .* kron(kxy, ones(Dx, Dy));
        end
        
    end

    methods (Access = public)
        
        % constructor
        function [obj] = SquareExponentialKernel(l)
            
            l = l(:);
            assert(all(l) > 0, 'Scale parameters must be positive.');
            obj.l = l;
        end
    end
end
