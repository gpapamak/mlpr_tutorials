classdef Kernel < handle
% Abstract class that models a kernel function.
%
% George Papamakarios, Nov 2015

    methods (Abstract, Access = protected)
        
        % evaluate kernel: one data matrix
        [S] = eval_x(obj, x)
        
        % evaluate kernel: two data matrices
        [S] = eval_xy(obj, x, y)
        
        % evaluate kernel and derivatives: one data matrix
        [S] = eval_x_d(obj, x)
        
        % evaluate kernel and derivatives: two data matrices
        [S] = eval_xy_d(obj, x, y)
    end
        
    methods (Access = protected)
        
        % helper function for checking derivatives with finite differences
        function [f, df] = f_df(obj, x, y, d)
            
            S = obj.eval_d(x, y);
            f = S(1, d + 1);
            df = S(2:end, d + 1);
        end
    end

    methods (Access = public)
        
        % evaluate kernel
        function [S] = eval(obj, x, y)
            
            if nargin < 3
                S = obj.eval_x(x);
            else
                S = obj.eval_xy(x, y);
            end
        end
        
        % evaluate kernel and derivatives
        function [S] = eval_d(obj, x, y)
            
            if nargin < 3
                S = obj.eval_x_d(x);
            else
                S = obj.eval_xy_d(x, y);
            end
        end
        
        % add kernels
        function [k] = plus(k1, k2)
            k = SumKernel(k1, k2);
        end
        
        % multiply kernels
        function [k] = mtimes(k1, k2)
            
            if ~isa(k1, 'Kernel')
                k = ScaleKernel(k1, k2);
            elseif ~isa(k2, 'Kernel')
                k = ScaleKernel(k2, k1);
            else
                k = ProductKernel(k1, k2);
            end
        end
        
        % perform a set of sanity checks
        function check(obj)
            
            D = 20;
            Nx = 100;
            Ny = 200;
            
            x = randn(D, Nx);
            y = randn(D, Ny);

            Sxx = obj.eval(x);
            Sxy = obj.eval(x, y);
            
            hasderivs = true;
            try
                Dxx = obj.eval_d(x);
                Dxy = obj.eval_d(x, y);
            catch exception
                if ismember(exception.identifier, {'GaussianProcesses:notImplemented', 'GaussianProcesses:nonDifferentiable'})
                    hasderivs = false;
                else
                    rethrow(exception)
                end
            end
            
            % check sizes
            fprintf('Matrix sizes... ');
            assert(isequal(size(Sxx), [Nx, Nx]), 'Matrix has wrong size.');
            assert(isequal(size(Sxy), [Nx, Ny]), 'Matrix has wrong size.');
            if hasderivs
                assert(isequal(size(Dxx), [Nx*(D+1), Nx*(D+1)]), 'Matrix has wrong size.');
                assert(isequal(size(Dxy), [Nx*(D+1), Ny*(D+1)]), 'Matrix has wrong size.');
            end
            fprintf('OK \n');
            
            % check symmetry
            fprintf('Symmetry... ');
            assert(isequal(Sxx, Sxx'), 'Matrix must be symmetric.');
            if hasderivs
                assert(isequal(Dxx, Dxx'), 'Matrix must be symmetric.');
            end
            fprintf('OK \n');
            
            % check positive semi-definiteness
            fprintf('Positive semi-definiteness... ');
            tol = 1.0e-9;
            assert(all(eig(Sxx) > -tol), 'Matrix must be PSD.');
            if hasderivs
                assert(all(eig(Dxx) > -tol), 'Matrix must be PSD.');
            end
            fprintf('OK \n');
            
            % check agreement of eval(x) and eval(x,y)
            fprintf('Consistency between eval(x) and eval(x,y)... ');
            tol = 1.0e-9;
            ii = randperm(Nx, floor(Nx/2));
            jj = randperm(Nx, floor(Nx/3));
            maxdiff = @(A,B) max(abs(A(:) - B(:)));
            assert(maxdiff(Sxx, obj.eval(x, x)) < tol, 'eval(x) and eval(x,y) are inconsistent.');
            assert(maxdiff(Sxx(ii,jj), obj.eval(x(:,ii), x(:,jj))) < tol, 'eval(x) and eval(x,y) are inconsistent.');
            if hasderivs
                di = [ii, Nx + D*kron(ii-1, ones(1,D)) + kron(ones(size(ii)), 1:D)];
                dj = [jj, Nx + D*kron(jj-1, ones(1,D)) + kron(ones(size(jj)), 1:D)];
                assert(maxdiff(Dxx, obj.eval_d(x, x)) < tol, 'eval_d(x) and eval_d(x,y) are inconsistent.');
                assert(maxdiff(Dxx(di,dj), obj.eval_d(x(:,ii), x(:,jj))) < tol, 'eval_d(x) and eval_d(x,y) are inconsistent.');
            end
            fprintf('OK \n');
            
            % check derivatives with finite differences
            if hasderivs
                fprintf('Derivatives... ');
                tol = 1.0e-9;
                assert(maxdiff(Sxx, Dxx(1:Nx, 1:Nx)) < tol, 'Function values in the derivative matrix are inconsistent.');
                assert(maxdiff(Sxy, Dxy(1:Nx, 1:Ny)) < tol, 'Function values in the derivative matrix are inconsistent.');
                xs = randn(D, 10);
                tol = 1.0e-5;
                for d = 0:D
                    err = checkgrad(xs, @(t) obj.f_df(t, zeros(D,1), d));
                    assert(err < tol, 'Derivatives are wrong.');
                end
                fprintf('OK \n');
            end
        end
        
    end
end
