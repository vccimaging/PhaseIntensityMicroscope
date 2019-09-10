clear;close all;
addpath('../matlab/');
addpath('../matlab/utils/');

% data paths
samples = {'blood', 'cheek', 'HeLa', 'MCF-7'};
dpaths = cellfun(@(x) ['../data/' x '/'], samples, 'un',0);

% parameters
pixel_size = 6.45; % [um]
z = 1.43e3; % [um]
scale_factor = pixel_size^2/z;

% loop for each sub-figure
A_final = cell(4,1);
phi_final = cell(4,1);
for i = 1:length(phi_final)
    disp(['Running sub-figure ' num2str(i) '/' num2str(length(phi_final))])
    dpath = dpaths{i};
    
    % read data
    if i == 1
        r = imread([dpath 'ref_2.tif']);
        s = imread([dpath 'cap_2.tif']);
    else
        r = imread([dpath 'ref.tif']);
        s = imread([dpath 'cap.tif']);
    end
    r = double(r)/2^14 * 255;
    s = double(s)/2^14 * 255;
    
    % our solver
    [A_ours, phi, wavefront_lap, I_warp] = cws(r, s);
    A_ours = sqrt(A_ours .* (1 + pixel_size/z*wavefront_lap));
    A_final{i} = A_ours; % amplitude
    phi = tilt_removal(phi*scale_factor); % OPD
    phi_final{i} = phi - min(phi(:));
end

