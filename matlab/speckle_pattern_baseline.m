function [A, phi, D] = speckle_pattern_baseline(I0, I)
% A:   amplitude
% phi: phase image
% D:   dark-field scattering

% window size
n = [3 3];
h = ones(n);

% calculate A2
A = meanfilt(I,h) ./ meanfilt(I0,h);
% A = medfilt2(A, n, 'symmetric');

% calculate D
D = stdfilt(I,h) ./ stdfilt(I0,h);
D = D ./ A;

% calculate local pixel shifts (wavefront slopes)
[w, ~] = imregdemons(I, A.*I0);

% integrate phi from w
phi = poisson_solver(w(:,:,1), w(:,:,2));
phi = phi - mean2(phi);

% square to get amplitude
A = sqrt(A);

end


function J = meanfilt(I, h)
%MEANFILT Local mean of image.

J = imfilter(I, h/sum(h(:)) , 'replicate');

end
