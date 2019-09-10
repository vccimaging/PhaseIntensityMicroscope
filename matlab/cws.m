function [A, phi, wavefront_lap, I_warp, objvaltotal] = cws(I0, I, opt)
% Simutanous intensity and wavefront recovery. Solve for:
%
% min_{A,phi}      || i(x+\nabla phi) - A i_0(x) ||_2^2              + ...
%            alpha || \nabla phi ||_1                                + ...
%            beta  ( || \nabla phi ||_2^2 + || \nabla^2 phi ||_2^2 ) + ...
%            gamma ( || \nabla A ||_1     + || \nabla^2 A ||_1 )     + ...
%            tau   ( || \nabla A ||_2^2   + || \nabla^2 A ||_2^2 )
% 
% Inputs:
% - I0: reference image
% - I:  measurement image
% - opt: solver options
%      - priors: prior weights [alpha beta gamma beta]
%                (default = [0.1,0.1,100,5])  
%      - iter: [total_alternating_iter A-update_iter phi-update_iter]
%                (default = [3 10 20])
%      - mu: ADMM parameters (default = [0.1 100])
%      - tol: phi-update tolerance stopping criteria (default = 0.05)
%      - L: padding size [pad_width pad_height]
%           (default nearest power of 2 of out_size, each in range [2, 256])
%      - isverbose: output verbose [A_verbose phi_verbose]
%           (default = [0 0])
% Outputs:
% - A: intensity
% - phi: wavefront
% - wavefront_lap: wavefront Laplacian
% - I_warp: warped image of I
% - objvaltotal: objective function
% 
% See also `cws_gpu_wrapper.m` for its GPU version.

% check images
if ~isa(I0,'double')
    I0 = double(I0);
end
if ~isa(I,'double')
    I = double(I);
end

if nargin < 3
    disp('Using default parameter settings ...')
    
    % tradeoff parameters
    alpha = 0.1;
    beta  = 0.1;
    gamma = 100;
    tau   = 5;
    
    % total number of alternation iterations
    iter = 3;
    
    % A-update parameters
    opt_A.isverbose = 0;
    opt_A.iter = 10;
    opt_A.mu_A = 1e-1;
    
    % phi-update parameters
    opt_phase.isverbose = 0;
    opt_phase.L = min((2.^ceil(log2(size(I0))) - size(I0)) / 2, 256); % suit to [1024 1024]
    opt_phase.mu = 100;
    opt_phase.iter = 20;
    
    % tolerance
    tol = 0.05;
else
    % check option, if not existed, use default parameters
    if ~isfield(opt,'priors')
        opt.priors = [0.1 0.1 100 5];
    end
    if ~isfield(opt,'iter')
        opt.iter = [3 10 20];
    end
    if ~isfield(opt,'mu')
        opt.mu = [0.1 100];
    end
    if ~isfield(opt,'tol')
        opt.tol = 0.05;
    end
    if ~isfield(opt,'L')
        opt.L = min((2.^ceil(log2(size(I0))) - size(I0)) / 2, 256); % suit to [1024 1024]
    end
    if ~isfield(opt,'isverbose')
        opt.isverbose = [0 0];
    end
    
    % parameters checkers
    opt.priors = max(0, opt.priors);
    if length(opt.priors) ~= 4
        error('length of opt.priors must equal to 4!');
    end
    opt.iter = round(opt.iter);
    if length(opt.iter) ~= 3
        error('length of opt.iter must equal to 3!');
    end
    opt.mu = max(0, opt.mu);
    if length(opt.mu) ~= 2
        error('length of opt.mu must equal to 2!');
    end
    opt.tol = max(0, opt.tol);
    if length(opt.tol) ~= 1
        error('length of opt.tol must equal to 1!');
    end
    opt.L = max(0, opt.L);
    if length(opt.L) ~= 2
        error('length of opt.L must equal to 2!');
    end
    
    % tradeoff parameters
    alpha = opt.priors(1);
    beta  = opt.priors(2);
    gamma = opt.priors(3);
    tau   = opt.priors(4);
    
    % total number of alternation iterations
    iter = opt.iter(1);
    
    % A-update parameters
    opt_A.isverbose = 0;
    opt_A.iter = opt.iter(2);
    opt_A.mu_A = opt.mu(1);
    
    % phi-update parameters
    opt_phase.isverbose = 0;
    opt_phase.L = opt.L;
    opt_phase.mu = opt.mu(2);
    opt_phase.iter = opt.iter(3);
    
    % tolerance
    tol = opt.tol;
end

% compute A-update parameters
gamma_new = gamma / mean2(abs(I0).^2);
tau_new   = tau   / mean2(abs(I0).^2);

% define operators in spatial domain
x_k = [0 0 0; -1 1 0; 0 0 0];
y_k = [0 -1 0; 0 1 0; 0 0 0];
l_k = [0 1 0; 1 -4 1; 0 1 0];

% define sizes
M = size(I0);
L = opt_phase.L;
N = M + 2*L;

% boundary mask
M1 = @(u) cat(3, u(L(1)+1:end-L(1),L(2)+1:end-L(2),1), ...
    u(L(1)+1:end-L(1),L(2)+1:end-L(2),2));

% specify boundary conditions
bc = 'symmetric';     % use DCT

% define operators
nabla  = @(phi) cat(3, imfilter(phi, x_k, bc), imfilter(phi, y_k, bc));
nabla2 = @(phi) imfilter(phi, l_k, bc);
nablaT = @(adj_phi) imfilter(adj_phi(:,:,1), rot90(x_k,2), bc) + ...
    imfilter(adj_phi(:,:,2), rot90(y_k,2), bc);
K      = @(x) cat(3, nabla(x), nabla2(x));
KT     = @(u) nablaT(u(:,:,1:2)) + nabla2(u(:,:,3));

% pre-calculated variables
[~, K_mat] = prepare_DCT_basis(M);

% inversion basis in DCT domain for A-update and phi-update
mat_A_hat = 1 + (tau_new + opt_A.mu_A) * K_mat;

% define proximal operator of A
prox_A = @(u) sign(u) .* max(abs(u) - gamma_new/(2*opt_A.mu_A), 0);

% define norms
norm2 = @(x) sum(abs(x(:)).^2);
norm1 = @(x) sum(abs(x(:)));

% define objective functions
obj_A     = @(A, b) norm2(A - b./I0) + gamma_new * norm1(K(A)) + tau_new * norm2(K(A));
obj_total = @(A, b, phi) norm2(I0.*A - b) + ...
    alpha * norm1(nabla(phi)) + beta * norm2(K(phi)) + ...
    gamma * norm1(K(A))       + tau  * norm2(K(A));

% define obj handle
disp_obj = @(s,i,objval) disp([s num2str(i) ', obj = ' num2str(objval,'%.4e')]);

% initialization
A   = ones(M);
phi = zeros(N);
I_warp = I;

% main loop
objvaltotal = zeros(iter,1);
disp_obj('iter = ', 0, obj_total(A, I_warp, phi));
for outer_loop = 1:iter
    % === A-update ===
    B    = zeros([size(A) 3]);
    zeta = zeros([size(A) 3]);
    disp('-- A-update')
    for i_A = 1:opt_A.iter
        % A-update
        A = idct(idct(dct(dct( I_warp./I0 + opt_A.mu_A*KT(B-zeta) ).').' ./ mat_A_hat).').';
        
        % pre-cache
        u = K(A) + zeta;
        
        % B-update
        B = prox_A(u);
        
        % zeta-update
        zeta = u - B;
        
        % show objective function
        if opt_A.isverbose
            disp_obj('---- iter = ', i_A, obj_A(A,I_warp));
        end
    end
    % median filter A
    A = medfilt2(A, [3 3], 'symmetric');
    
    % === phi-update ===
    disp('-- phi-update')
    img = cat(3, A.*I0, I_warp);
    [~, Delta_phi, I_warp] = phase_update_ADMM(img, alpha, beta, opt_phase);
    
    disp(['-- mean(|\Delta\phi|) = ' num2str(mean(abs(Delta_phi(:))),'%.3e')])
    if mean(abs(Delta_phi(:))) < tol
        disp('-- mean(|\Delta\phi|) too small; quit');
        break;
    end
    phi = phi + Delta_phi;
    
    % === records ===
    objvaltotal(outer_loop) = obj_total(A, I_warp, phi);
    disp_obj('iter = ', outer_loop, objvaltotal(outer_loop));
end

% compute wavefront Laplacian
wavefront_lap = nabla2(phi);
wavefront_lap = M1(cat(3,wavefront_lap,wavefront_lap));
wavefront_lap = wavefront_lap(:,:,1);

% median filtering phi
phi = medfilt2(phi, [3 3], 'symmetric');

% return phi
phi = M1(cat(3,phi,phi));
phi = phi(:,:,1);
phi = phi - mean2(phi);

return;

    function [x, x_full, I_warp] = phase_update_ADMM(img, alpha, beta, opt)
        
        M = [size(img,1) size(img,2)];
        L = opt.L;
        N = M + 2*L;
        
        % boundary mask
        M1 = @(u) cat(3, u(L(1)+1:end-L(1),L(2)+1:end-L(2),1), ...
            u(L(1)+1:end-L(1),L(2)+1:end-L(2),2));
        [lap_mat, ~] = prepare_DCT_basis(N);
        mat_x_hat = (opt.mu+beta)*lap_mat + beta*lap_mat.^2;
        mat_x_hat(1) = 1;
        
        % initialization
        x    = zeros(N);
        u    = zeros([N 2]);
        zeta = zeros([N 2]);
        
        % get the matrices
        [gt, gx, gy] = partial_deriv(img);
        
        % pre-compute
        mu  = opt.mu;
        gxy = gx.*gy;
        gxx = gx.^2;
        gyy = gy.^2;
        
        % store in memory at run-time
        denom = mu*(gxx + gyy + mu);
        A11 = (gyy + mu) ./ denom;
        A12 = -gxy ./ denom;
        A22 = (gxx + mu) ./ denom;
        
        % proximal algorithm
        objval = zeros(opt.iter,1);
        time   = zeros(opt.iter,1);
        
        % the loop
        tic;
        for k = 1:opt.iter
            
            % x-update
            if k > 1
                x = idct(idct(dct(dct(mu*nablaT(u-zeta)).').' ./ mat_x_hat).').';
            end
            
            % pre-compute nabla_x
            nabla_x = nabla(x);
            
            % u-update
            u = nabla_x + zeta;
            u_temp = M1(u);
            
            % R2 LASSO (in practice, difference between the two solutions
            % are subtle; for speed's sake, the simple version is
            % recommended)
            [w_opt_x,w_opt_y] = R2LASSO(gx,gy,gt,u_temp(:,:,1),u_temp(:,:,2),alpha,mu,A11,A12,A22,'complete');
            
            % update u
            u(L(1)+1:end-L(1), L(2)+1:end-L(2), 1) = w_opt_x;
            u(L(1)+1:end-L(1), L(2)+1:end-L(2), 2) = w_opt_y;
            
            % zeta-update
            zeta = zeta + nabla_x - u;
            
            % record
            if opt.isverbose
                if k == 1
                    % define operators
                    G = @(u) gx.*u(:,:,1) + gy.*u(:,:,2);
                    
                    % define objective function
                    obj = @(phi) norm2(G(M1(nabla(phi)))+gt) + ...
                        alpha *  norm1(nabla(phi)) + ...
                        beta  * (norm2(nabla(phi)) + norm2(nabla2(phi)));
                end
                objval(k) = obj(x);
                time(k)   = toc;
                disp(['---- ADMM iter: ' num2str(k) ', obj = ' num2str(objval(k),'%.4e')])
            end
        end
        
        % compute the warped image by Taylor expansion
        w = M1(nabla(x));
        I_warp = gx.*w(:,:,1) + gy.*w(:,:,2) + img(:,:,2);
        
        % return masked x
        x = x - mean2(x);
        x_full = x;
        x = M1(cat(3,x,x));
        x = x(:,:,1);
    end
end


function [It,Ix,Iy] = partial_deriv(images)

% derivative kernel
h = [1 -8 0 8 -1]/12;

% temporal gradient
It = images(:,:,2) - images(:,:,1);

% First compute derivative then warp
Ix = imfilter(images(:,:,2), h,  'replicate');
Iy = imfilter(images(:,:,2), h', 'replicate');

end


function [lap_mat, K_mat] = prepare_DCT_basis(M)
% Prepare DCT basis of size M:
% lap_mat: \nabla^2
% K_mat:   \nabla^2 + \nabla^4

H = M(1);
W = M(2);
[x_coord,y_coord] = meshgrid(0:W-1,0:H-1);
lap_mat = 4 - 2*cos(pi*x_coord/W) - 2*cos(pi*y_coord/H);
K_mat   = lap_mat .* (lap_mat + 1);

end


function [x, y] = R2LASSO(a,b,c,ux,uy,alpha,mu,A11,A12,A22,option)
switch option
    case 'simple'
        [x, y] = R2LASSO_simple(a,b,c,ux,uy,alpha,mu,A11,A12,A22);
    case 'complete'
        [x, y] = R2LASSO_complete(a,b,c,ux,uy,alpha,mu,A11,A12,A22);
    otherwise
        error('invalid option for R2LASSO solver!')
end
end


function [x, y] = R2LASSO_simple(a,b,c,ux,uy,alpha,mu,A11,A12,A22)
% This function attempts to solve the R2 LASSO problem, in a fast but not accurate sense:
% min (a x + b y + c)^2 + \mu [(x - ux)^2 + (y - uy)^2] + \alpha (|x| + |y|)
% x,y
%
% See also R2LASSO_complete for a more accurate but slower one.
%
% A11, A12 and A22 can be computed as:
% A_tmp = 1/(mu*(a*a + b*b + mu)) .* [b*b + mu  -a*b; -a*b  a*a + mu];
% A11 = A_tmp(1);
% A12 = A_tmp(2);
% A22 = A_tmp(4);

if ~exist('A11','var') || ~exist('A12','var') || ~exist('A22','var')
    denom = 1 ./ ( mu * (a.*a + b.*b + mu) );
    A11 = denom * (b.*b + mu);
    A12 = -a.*b .* denom;
    A22 = denom .* (a.*a + mu);
end

% 1. get (x0, y0)

% temp
tx = mu*ux - a.*c;
ty = mu*uy - b.*c;

% l2 minimum
x0_x = A11.*tx + A12.*ty;
x0_y = A12.*tx + A22.*ty;

% 2. get the sign of optimal
sign_x = sign(x0_x);
sign_y = sign(x0_y);

% 3. get the optimal
x = x0_x - alpha/2 * (A11 .* sign_x + A12 .* sign_y);
y = x0_y - alpha/2 * (A12 .* sign_x + A22 .* sign_y);

% 4. check sign and map to the range
x( sign(x) ~= sign_x ) = 0;
y( sign(y) ~= sign_y ) = 0;

end


function [x, y] = R2LASSO_complete(a,b,c,ux,uy,alpha,mu,A11,A12,A22)
% This function attempts to solve the R2 LASSO problem in a most accurate sense:
% min (a x + b y + c)^2 + \mu [(x - ux)^2 + (y - uy)^2] + \alpha (|x| + |y|)
% x,y
%
% See also R2LASSO_simple for a faster but less accurate one.
%
% A11, A12 and A22 can be computed as:
% A_tmp = 1/(mu*(a*a + b*b + mu)) .* [b*b + mu  -a*b; -a*b  a*a + mu];
% A11 = A_tmp(1);
% A12 = A_tmp(2);
% A22 = A_tmp(4);

if ~exist('A11','var') || ~exist('A12','var') || ~exist('A22','var')
    denom = 1 ./ ( mu * (a.*a + b.*b + mu) );
    A11 = denom .* (b.*b + mu);
    A12 = -a.*b .* denom;
    A22 = denom .* (a.*a + mu);
end

% temp
tx = mu*ux - a.*c;
ty = mu*uy - b.*c;

% l2 minimum
x0_x = A11.*tx + A12.*ty;
x0_y = A12.*tx + A22.*ty;

% x1 (R^n) [x1 0] & x2 (R^n) [0 x2]
% if (tx > 0)
%     x1 = (tx - alpha/2) ./ (a.*a + mu);
% else
%     x1 = (tx + alpha/2) ./ (a.*a + mu);
% end
% if (ty > 0)
%     x2 = (ty - alpha/2) ./ (b.*b + mu);
% else
%     x2 = (ty + alpha/2) ./ (b.*b + mu);
% end
x1         = (tx + alpha/2) ./ (a.*a + mu);
tmp = (tx - alpha/2) ./ (a.*a + mu);
ind = tx > 0;
x1(ind) = tmp(ind);

x2         = (ty + alpha/2) ./ (b.*b + mu);
tmp = (ty - alpha/2) ./ (b.*b + mu);
ind = ty > 0;
x2(ind) = tmp(ind);

% x3 (R^2n)
% tx = alpha/2 * (A11 + A12);
% ty = alpha/2 * (A12 + A22);
% if (tx*x0_x + ty*x0_y) > 0
%     x3_x = x0_x - tx;
%     x3_y = x0_y - ty;
% else
%     x3_x = x0_x + tx;
%     x3_y = x0_y + ty;
% end
tx = alpha/2 * (A11 + A12);
ty = alpha/2 * (A12 + A22);
x3_x = x0_x + tx;
x3_y = x0_y + ty;
tmp_x = x0_x - tx;
tmp_y = x0_y - ty;
ind = (tx.*x0_x + ty.*x0_y) > 0;
x3_x(ind) = tmp_x(ind);
x3_y(ind) = tmp_y(ind);

% x4 (R^2n)
% tx = alpha/2 * (A11 - A12);
% ty = alpha/2 * (A12 - A22);
% if (tx*x0_x + ty*x0_y) > 0
%     x4_x = x0_x - tx;
%     x4_y = x0_y - ty;
% else
%     x4_x = x0_x + tx;
%     x4_y = x0_y + ty;
% end
tx = alpha/2 * (A11 - A12);
ty = alpha/2 * (A12 - A22);
x4_x = x0_x + tx;
x4_y = x0_y + ty;
tmp_x = x0_x - tx;
tmp_y = x0_y - ty;
ind = (tx.*x0_x + ty.*x0_y) > 0;
x4_x(ind) = tmp_x(ind);
x4_y(ind) = tmp_y(ind);

% cost functions
cost = inf([size(a) 5]);

% x1
t1 = a.*x1 + c;
tx = x1 - ux;
cost(:,:,1) = t1.*t1 + mu*(tx.*tx + uy.*uy) + alpha*abs(x1);

% x2
t1 = b.*x2 + c;
ty = x2 - uy;
cost(:,:,2) = t1.*t1 + mu*(ux.*ux + ty.*ty) + alpha*abs(x2);

% x3
t1 = a.*x3_x + b.*x3_y + c;
tx = x3_x - ux;
ty = x3_y - uy;
cost(:,:,3) = t1.*t1 + mu*(tx.*tx + ty.*ty) + alpha*(abs(x3_x) + abs(x3_y));

% x4
t1 = a.*x4_x + b.*x4_y + c;
tx = x4_x - ux;
ty = x4_y - uy;
cost(:,:,4) = t1.*t1 + mu*(tx.*tx + ty.*ty) + alpha*(abs(x4_x) + abs(x4_y));

% x5
cost(:,:,5) = c.*c + mu*(ux.*ux + uy.*uy); % [0 0] solution

% find minimum
cost_min = min(cost, [], 3);
x_can = cat(3, x1, zeros(size(a)), x3_x, x4_x, zeros(size(a)));
y_can = cat(3, zeros(size(a)), x2, x3_y, x4_y, zeros(size(a)));
ind = cost == cost_min;

% return
x = sum(ind .* x_can, 3);
y = sum(ind .* y_can, 3);

end
