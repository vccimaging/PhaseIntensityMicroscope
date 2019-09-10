clc;clear;close all;
addpath('./utils/');

% datapath
datapath = '../data/MLA-150-7AR-M/';

% read image
ref = double(imread([datapath 'ref.tif']));
cap = double(imread([datapath 'cap.tif']));

% set active area
M = [992 992];

% crop size
S = (size(ref)-M)/2;

% crop data
ref = ref(1+S(1):end-S(1),1+S(2):end-S(2));
cap = cap(1+S(1):end-S(1),1+S(2):end-S(2));

% normalize images
norm_img = @(x) double(uint8( 255 * x ./ max( max(ref(:)), max(cap(:)) ) ));
ref = norm_img(ref);
cap = norm_img(cap);

% set gpu parameters
opt.priors = [0.1 0.1 100 5];
opt.iter   = [3 10 20];
opt.mu     = [0.1 100];
opt.tol    = 0.05;
opt.L      = min((2.^ceil(log2(M)) - M) / 2, 256);
opt.size   = M;
opt.isverbose = 1;

% run cpu algorithm
tic;
[A_cpu, phi_cpu, ~, ~, ~] = cws(ref, cap, opt);
toc

% run gpu algorithm
[A_gpu, phi_gpu] = cws_gpu_wrapper(ref, cap, opt);

% mean normalized phase
phi_gpu = phi_gpu - mean2(phi_gpu);

% show results
figure;     imshow([A_cpu A_gpu A_cpu-A_gpu],[]);
title('A: CPU / GPU / Difference');
disp(['A: max diff = ' num2str(norm(A_gpu(:) - A_cpu(:),'inf'))]);
figure;     imshow([phi_cpu phi_gpu phi_cpu-phi_gpu],[]);
title('phi: CPU / GPU / Difference');
disp(['phi: max diff = ' num2str(norm(phi_gpu(:) - phi_cpu(:),'inf'))]);
figure;mesh(phi_cpu - phi_gpu)
title('phi: CPU / GPU / Difference');
