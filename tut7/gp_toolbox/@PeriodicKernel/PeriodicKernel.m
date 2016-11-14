classdef PeriodicKernel < Kernel
% Periodic kernel: produces smooth periodic functions.
%
% George Papamakarios, Nov 2015

    properties (SetAccess = private, GetAccess = public)
        
        l = []
        T = []
        
    end
    
    methods (Access = protected)
               
        % evaluate kernel: one data matrix
        function [S] = eval_x(obj, x)
            
            % number of datapoints
            Nx = size(x, 2);
            
            % scale inputs
            if isscalar(obj.T)
                x = x / (obj.T / pi);
            else
                x = x ./ ((obj.T / pi) * ones(1, Nx));
            end
            
            % evaluate kernel
            x = repmat(permute(x, [3 2 1]), Nx, 1, 1);
            A = sin(x - permute(x, [2 1 3]));
            if isscalar(obj.l)
                A = A / obj.l;
            else
                A = A ./ repmat(permute(obj.l, [3 2 1]), Nx, Nx, 1);
            end
            S = exp(-sum(A.^2, 3) / 2);
            
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
            if isscalar(obj.T)
                x = x / (obj.T / pi);
                y = y / (obj.T / pi);
            else
                x = x ./ ((obj.T / pi) * ones(1, Nx));
                y = y ./ ((obj.T / pi) * ones(1, Ny));
            end
            
            % evaluate kernel
            x = repmat(permute(x, [2 3 1]), 1, Ny, 1);
            y = repmat(permute(y, [3 2 1]), Nx, 1, 1);
            A = sin(x - y);
            if isscalar(obj.l)
                A = A / obj.l;
            else
                A = A ./ repmat(permute(obj.l, [3 2 1]), Nx, Ny, 1);
            end
            S = exp(-sum(A.^2, 3) / 2);
        end
        
        % evaluate kernel and derivatives: one data matrix
        function [S] = eval_x_d(obj, x)
            
            [Dx, Nx] = size(x);
            kxx = eval_x(obj, x);
            
            if isscalar(obj.T)
                objT = ones(Dx, 1) * obj.T;
            else
                objT = obj.T;
            end
            if isscalar(obj.l)
                objl = ones(Dx, 1) * obj.l;
            else
                objl = obj.l;
            end
            
            x = x ./ ((objT / pi) * ones(1, Nx));
            
            xmx = x(:) * ones(1, Nx) - repmat(x, Nx, 1);
            sinxmx = sin(xmx);
            cosxmx = cos(xmx);
            sincosxmx = pi ./ repmat(objT .* objl.^2, Nx, Nx) .* sinxmx .* cosxmx;
            Sdxx = -sincosxmx .* kron(kxx, ones(Dx, 1));
            diag_Sdxdx = pi^2 ./ repmat(objT.^2 .* objl.^2, Nx, Nx) .* (cosxmx.^2 - sinxmx.^2);
            
            S = zeros(Nx*(Dx + 1), 'like', x);
            S(1:Nx, 1:Nx) = kxx;
            S(Nx+1:end, 1:Nx) = Sdxx;
            S(1:Nx, Nx+1:end) = Sdxx';
            S(Nx+1:end, Nx+1:end) = (kron(diag_Sdxdx, ones(1, Dx)) .* repmat(eye(Dx), Nx, Nx) + kron(sincosxmx, ones(1, Dx)) .* kron(sincosxmx', ones(Dx, 1))) .* kron(kxx, ones(Dx, Dx));
            
            S = (S + S') / 2;
        end 
        
        % evaluate kernel and derivatives: two data matrices
        function [S] = eval_xy_d(obj, x, y)
            
            [Dx, Nx] = size(x);
            [Dy, Ny] = size(y);
            assert(Dx == Dy, 'Dimensionalities don''t match.');
            kxy = eval_xy(obj, x, y);
            
            if isscalar(obj.T)
                objT = ones(Dx, 1) * obj.T;
            else
                objT = obj.T;
            end
            if isscalar(obj.l)
                objl = ones(Dx, 1) * obj.l;
            else
                objl = obj.l;
            end
            
            x = x ./ ((objT / pi) * ones(1, Nx));
            y = y ./ ((objT / pi) * ones(1, Ny));
            
            xmyv = x(:) * ones(1, Ny) - repmat(y, Nx, 1);
            sinxmyv = sin(xmyv);
            cosxmyv = cos(xmyv);
            sincosxmyv = pi ./ repmat(objT .* objl.^2, Nx, Ny) .* sinxmyv .* cosxmyv;
            Sdxyv = -sincosxmyv .* kron(kxy, ones(Dx, 1));
                        
            xmyh = repmat(x', 1, Ny) - ones(Nx, 1) * y(:)';
            sinxmyh = sin(xmyh);
            cosxmyh = cos(xmyh);
            sincosxmyh = pi ./ repmat(objT' .* objl'.^2, Nx, Ny) .* sinxmyh .* cosxmyh;
            Sdxyh = sincosxmyh .* kron(kxy, ones(1, Dy));
            
            diag_Sdxdy = pi^2 ./ repmat(objT.^2 .* objl.^2, Nx, Ny) .* (cosxmyv.^2 - sinxmyv.^2);
            
            S = zeros(Nx*(Dx + 1), Ny*(Dy + 1), 'like', x);
            S(1:Nx, 1:Ny) = kxy;
            S(Nx+1:end, 1:Ny) = Sdxyv;
            S(1:Nx, Ny+1:end) = Sdxyh;
            S(Nx+1:end, Ny+1:end) = (kron(diag_Sdxdy, ones(1, Dy)) .* repmat(eye(Dx,Dy), Nx, Ny) - kron(sincosxmyv, ones(1, Dy)) .* kron(sincosxmyh, ones(Dx, 1))) .* kron(kxy, ones(Dx, Dy));
            
        end
        
    end

    methods (Access = public)
        
        % constructor
        function [obj] = PeriodicKernel(l, T)
            
            l = l(:);
            T = T(:);
            assert(all(l) > 0, 'Scale parameters must be positive.');
            assert(all(T) > 0, 'Periods must be positive.');
            obj.l = l;
            obj.T = T;
        end
    end
end
