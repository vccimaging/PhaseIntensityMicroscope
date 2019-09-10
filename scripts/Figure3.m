clc;clear;close all;
addpath('../matlab/');
addpath('../matlab/utils/');

% input data path
dpath = '../data/blood/';
r = imread([dpath 'ref_1.tif']);
s = imread([dpath 'cap_1.tif']);
n = 2; % we are doing OPD here

% parameters
pixel_size = 6.45; % [um]
z = 1.43e3; % [um]
scale_factor = pixel_size^2/z;
map = jet(256);

% read data
r = double(r)/2^14 * 255;
s = double(s)/2^14 * 255;


%% Methods

%%% Slope-tracking
[w, ~] = imregdemons(s, r, 200);
phi_tracking = poisson_solver(w(:,:,1), w(:,:,2));
phi_tracking = phi_tracking - mean2(phi_tracking);
phi_tracking = tilt_removal(phi_tracking/(n-1)*scale_factor);


%%% Wang et al.
opt.isverbose = 0;
opt.L = {[20 20]};
opt.mu = 100;
opt.iter = 30;
warping_iter = 2;
beta = 1;
phi_wang = main_wavefront_solver(cat(3, r, s), beta, opt, warping_iter);
phi_wang = tilt_removal(phi_wang/(n-1)*scale_factor);


%%% Baseline
[A_base, phi_base, D_base] = speckle_pattern_baseline(r, s);
phi_base = tilt_removal(phi_base/(n-1)*scale_factor);


%%% Ours
opt_cws.priors = [0.5 0.5 100 5];
[A_ours, phi, wavefront_lap, I_warp] = cws(r, s, opt_cws);
A_ours = A_ours .* (1 + pixel_size/z*wavefront_lap);
A_ours = sqrt(A_ours);
I = A_ours; % amplitude
phi = tilt_removal(phi/(n-1)*scale_factor);

% denoise a little bit ...
phi = medfilt2(phi, [3 3], 'symmetric');


%% Show results

% normalize to start from 0
phi_tracking = phi_tracking - min(phi_tracking(:));
phi_wang     = phi_wang - min(phi_wang(:));
phi_base     = phi_base - min(phi_base(:));
phi          = phi - min(phi(:));

% show results
figure;imshow(phi_tracking, []);
axis tight ij;colormap(map);pause(0.2);
title('Berto et al. 2017');

figure;imshow(phi_wang, []);
axis tight ij;colormap(map);pause(0.2)
title('Wang et al. 2017');

figure;imshow(phi_base, []);
axis tight ij;colormap(map);pause(0.2)
title('Berujon et al. 2015');

figure;imshow(phi, []);
axis tight ij;colormap(map);pause(0.2)
title('Ours');


%% Cross-sections

% function handles
get_c = @(phi) phi(191,390:550)';
min_n = @(x) x - min(x(:));

% get the maxs
max(phi_tracking(:))
max(phi_wang(:))
max(phi_base(:))
max(phi(:))

% get cross-sections
t_tracking = get_c(phi_tracking);
t_wang = get_c(phi_wang);
t_base = get_c(phi_base);
t_ours = get_c(phi);

% show plots
figure;
plot([min_n(t_tracking) min_n(t_wang) min_n(t_base) min_n(t_ours)],'LineWidth',2);
axis tight;
legend('Berto et al. 2017','Wang et al. 2017','Berujon et al. 2015','Ours');

