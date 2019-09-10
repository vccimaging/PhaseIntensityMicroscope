clc;clear;close all;
addpath('../matlab/');
addpath('../matlab/utils/');

% input data path
dpath = '../data/MLA-150-7AR-M/';
r = imread([dpath 'ref.tif']);
s = imread([dpath 'cap.tif']);
n = 1.46; % refractive index

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

% for MLA
% set parameters
x1=253;  y1=360;
x2=670;  y2=585;
x3=1082; y3=800;

% set two points for cross-sectioning
xx = [x2 x3];
yy = [y2 y3];

% get the indexes
a = (yy(2)-yy(1))/(xx(2)-xx(1));
b = yy(1) - a*xx(1);
x = xx(1):xx(2);
y = round(a*x+b);
ind = sub2ind(size(phi),y,x);

% show the points
figure; imshow(phi, []); axis tight;
hold on;        line([xx(1) xx(2)],[yy(1) yy(2)],'Color',[1 0 0]);

% get cross-section
get_c = @(phi) phi(ind)';

% get the maxs
max(phi_tracking(:))
max(phi_wang(:))
max(phi_base(:))
max(phi(:))
max_phi = round(max(phi(:)),2);

% get cross-sections
t_tracking = get_c(phi_tracking);
t_wang = get_c(phi_wang);
t_base = get_c(phi_base);
t_ours = get_c(phi);

% min-normalized
t_tracking = t_tracking - min(t_tracking);
t_wang = t_wang - min(t_wang);
t_base = t_base - min(t_base);
t_ours = t_ours - min(t_ours);

% x coordinates
tt = 1:length(t_tracking);
tt = tt - mean(tt);
tt = tt * pixel_size/20 / 0.9;

% show plots
figure;     plot(tt, [t_tracking t_wang t_base t_ours],'LineWidth',2);
axis tight;
legend('Berto et al. 2017','Wang et al. 2017','Berujon et al. 2015','Ours');


%% Comparison with Zygo data

% read data
[dat, xl, yl] = LoadMetroProData([dpath 'Zygo.dat']);
dat(isnan(dat)) = mean(dat(~isnan(dat)));

% tilt removal
dat = tilt_removal(dat*1e6);

% show raw data
figure;     imshow(dat,[],'i','f');      axis tight on
colormap(map);colorbar;
title('Zygo raw');

% correspond region
xx = [188 222];
yy = [157 155];

% get the indices
a = (yy(2)-yy(1))/(xx(2)-xx(1));
b = yy(1) - a*xx(1);
x = xx(1):xx(2);
y = round(a*x+b);
ind = sub2ind(size(dat),y,x);
t_gt = dat(ind);
t_gt = t_gt - min(t_gt);

% x coordinates
tt = 1:length(t_tracking);
tt = tt - mean(tt);
tt = tt * pixel_size/20 / 0.9; % compensation for small misalignment

% x coordinates
tt = imresize(tt, [1 numel(t_gt)]);

% plot
figure;     plot(tt, t_gt, 'o', 'LineWidth',2);
title('Zygo plot')

% calculate RMS
t_gt = imresize(t_gt', [numel(t_tracking) 1]);
calc_rms = @(x) sqrt(mean(abs(x - t_gt).^2));

% show RMS for each method
disp('RMS is:');
disp(['Berto et al. 2017: ' num2str(calc_rms(t_tracking)) ' um']);
disp(['Wang et al. 2017: ' num2str(calc_rms(t_wang)) ' um']);
disp(['Berujon et al. 2015: ' num2str(calc_rms(t_base)) ' um']);
disp(['Ours: ' num2str(calc_rms(t_ours)) ' um']);

