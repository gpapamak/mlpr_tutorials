classdef NeuralNetKernel < Kernel
% Neural net kernel, see:
%
% C. Williams, "Computation with Infinite Neural Networks", Neural 
% Computation, vol. 10, num. 5, pp. 1203-1216, 1998.
%
% George Papamakarios, Nov 2015

    properties (SetAccess = private, GetAccess = public)
        
        vw = []
        vb = []
        
    end
    
    methods (Access = protected)
               
        % evaluate kernel: one data matrix
        function [S] = eval_x(obj, x)
            
            [Dx, Nx] = size(x);
            
            % construct parameter covariance
            C = obj.vw * eye(Dx + 1);
            C(end, end) = obj.vb;
            
            % append ones
            x = [x; ones(1, Nx)];
            
            % evaluate kernel
            A1 = 2 * x' * C * x;
            A2 = diag(A1) * ones(1, Nx) + 1;
            S = (2 / pi) * asin(A1 ./ sqrt(A2 .* A2'));
            
            % for numerical stability
            S = (S + S') / 2;
        end
        
        % evaluate kernel: two data matrices
        function [S] = eval_xy(obj, x, y)
            
            [Dx, Nx] = size(x);
            [Dy, Ny] = size(y);
            assert(Dx == Dy, 'Dimensionalities don''t match.');
            
            % construct parameter covariance
            C = obj.vw * eye(Dx + 1, Dy + 1);
            C(end, end) = obj.vb;
            
            % append ones
            x = [x; ones(1, Nx)];
            y = [y; ones(1, Ny)];
            
            % evaluate kernel
            Cx = C * x;
            Cy = C * y;
            Ax = 2 * sum(x .* Cx, 1)' * ones(1, Ny) + 1;
            Ay = 2 * ones(Nx, 1) * sum(y .* Cy, 1) + 1;
            Axy = 2 * x' * Cy;
            S = (2 / pi) * asin(Axy ./ sqrt(Ax .* Ay));
        end
        
        % evaluate kernel and derivatives: one data matrix
        function [S] = eval_x_d(obj, x)
            
            [Dx, Nx] = size(x);
            
            C = obj.vw * eye(Dx + 1);
            C(end, end) = obj.vb;
            
            xbar = [x; ones(1, Nx)];
            
            Cx = 2 * C * xbar;
            xCy = xbar' * Cx;
            xCx = diag(xCy) * ones(1, Nx);
            
            denom = sqrt((1 + xCx) .* (1 + xCx'));
            z = xCy ./ denom;
            
            Axb = Cx(1:end-1, :);
            dzdx_1 = repmat(Axb, Nx, 1) ./ kron(denom, ones(Dx, 1));
            dzdx_2 = kron(xCy .* (1+xCx') ./ denom.^3, ones(Dx, 1)) .* (Axb(:) * ones(1, Nx));
            dzdx = dzdx_1 - dzdx_2;
            
            d2zdxdy_1 = repmat(2 * C(1:end-1, 1:end-1), Nx, Nx) ./ kron(denom, ones(Dx, Dx));
            d2zdxdy_2 = kron(repmat(Axb, Nx, 1), ones(1, Dx)) .* (ones(Nx*Dx, 1) * Axb(:)') .* kron(1 + xCx, ones(Dx, Dx));
            d2zdxdy_3 = (Axb(:) * ones(1, Nx*Dx)) .* kron(repmat(Axb', 1, Nx), ones(Dx, 1)) .* kron(1 + xCx', ones(Dx, Dx));
            d2zdxdy_4 = (Axb(:) * ones(1, Nx*Dx)) .* (ones(Nx*Dx, 1) * Axb(:)') .* kron(-xCy, ones(Dx, Dx));
            d2zdxdy = d2zdxdy_1 - (d2zdxdy_2 + d2zdxdy_3 + d2zdxdy_4) ./ kron(denom.^3, ones(Dx, Dx));
            
            buf = sqrt(1 - z.^2);
            dkdz = (2 / pi) ./ buf;
            d2kdz2 = (2 / pi) * z ./ (buf .^ 3);
            
            Sdxx = kron((2 / pi) ./ sqrt(1 - z.^2), ones(Dx, 1)) .* dzdx;
                        
            S = zeros(Nx*(Dx + 1), 'like', x);
            S(1:Nx, 1:Nx) = (2 / pi) * asin(z);
            S(Nx+1:end, 1:Nx) = Sdxx;
            S(1:Nx, Nx+1:end) = Sdxx';
            S(Nx+1:end, Nx+1:end) = kron(d2kdz2, ones(Dx,Dx)) .* kron(dzdx, ones(1,Dx)) .* kron(dzdx', ones(Dx,1)) + kron(dkdz, ones(Dx,Dx)) .* d2zdxdy;
            
            S = (S + S') / 2;
        end
        
        % evaluate kernel and derivatives: two data matrices
        function [S] = eval_xy_d(obj, x, y)
            
            [Dx, Nx] = size(x);
            [Dy, Ny] = size(y);
            assert(Dx == Dy, 'Dimensionalities don''t match.');
            
            C = obj.vw * eye(Dx + 1, Dy + 1);
            C(end, end) = obj.vb;
            
            xbar = [x; ones(1, Nx)];
            ybar = [y; ones(1, Ny)];
            
            Cx = 2 * C * xbar;
            Cy = 2 * C * ybar;
            xCy = xbar' * Cy;
            xCx = sum(xbar .* Cx, 1)' * ones(1, Ny);
            yCy = ones(Nx, 1) * sum(ybar .* Cy, 1);
                        
            denom = sqrt((1 + xCx) .* (1 + yCy));
            z = xCy ./ denom;
            
            Axb = Cx(1:end-1, :);
            Ayb = Cy(1:end-1, :);
            
            dzdx_1 = repmat(Ayb, Nx, 1) ./ kron(denom, ones(Dx, 1));
            dzdx_2 = kron(xCy .* (1+yCy) ./ denom.^3, ones(Dx, 1)) .* (Axb(:) * ones(1, Ny));
            dzdx = dzdx_1 - dzdx_2;
            
            dzdy_1 = repmat(Axb', 1, Ny) ./ kron(denom, ones(1, Dy));
            dzdy_2 = kron(xCy .* (1+xCx) ./ denom.^3, ones(1, Dy)) .* (ones(Nx, 1) * Ayb(:)');
            dzdy = dzdy_1 - dzdy_2;
            
            d2zdxdy_1 = repmat(2 * C(1:end-1, 1:end-1), Nx, Ny) ./ kron(denom, ones(Dx, Dy));
            d2zdxdy_2 = kron(repmat(Ayb, Nx, 1), ones(1, Dy)) .* (ones(Nx*Dx, 1) * Ayb(:)') .* kron(1 + xCx, ones(Dx, Dy));
            d2zdxdy_3 = (Axb(:) * ones(1, Ny*Dy)) .* kron(repmat(Axb', 1, Ny), ones(Dx, 1)) .* kron(1 + yCy, ones(Dx, Dy));
            d2zdxdy_4 = (Axb(:) * ones(1, Ny*Dy)) .* (ones(Nx*Dx, 1) * Ayb(:)') .* kron(-xCy, ones(Dx, Dy));
            d2zdxdy = d2zdxdy_1 - (d2zdxdy_2 + d2zdxdy_3 + d2zdxdy_4) ./ kron(denom.^3, ones(Dx, Dy));
            
            buf = sqrt(1 - z.^2);
            dkdz = (2 / pi) ./ buf;
            d2kdz2 = (2 / pi) * z ./ (buf .^ 3);
                                    
            S = zeros(Nx*(Dx + 1), Ny*(Dy + 1), 'like', x);
            S(1:Nx, 1:Ny) = (2 / pi) * asin(z);
            S(Nx+1:end, 1:Ny) = kron(dkdz, ones(Dx, 1)) .* dzdx;
            S(1:Nx, Ny+1:end) = kron(dkdz, ones(1, Dy)) .* dzdy;
            S(Nx+1:end, Ny+1:end) = kron(d2kdz2, ones(Dx,Dy)) .* kron(dzdx, ones(1,Dy)) .* kron(dzdy, ones(Dx,1)) + kron(dkdz, ones(Dx,Dy)) .* d2zdxdy;
        end
        
    end

    methods (Access = public)
        
        % constructor
        function [obj] = NeuralNetKernel(vw, vb)
            
            check = @(t) isscalar(t) && t >= 0;
            assert(check(vw) && check(vb), 'Kernel parameters must be non-negative scalars.');
            
            obj.vw = vw;
            obj.vb = vb;
        end
    end
end
