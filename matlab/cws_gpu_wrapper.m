function [A, phi] = cws_gpu_wrapper(I0, I, opt)
% This is a MATLAB wrapper for GPU solver `cws`.
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
%      - size: output size (this is unavailable in the CPU version)
% 
% Outputs:
% - A: intensity
% - phi: wavefront
% 
% See also `cws.m` for its CPU version, and `cpu_gpu_comparison` as demo.

% get cws solver
tpath = mfilename('fullpath');
if ismac || isunix
    cws_path = [tpath(1:end-23) '/cuda/bin/cws'];
elseif ispc
    cws_path = ['"' tpath(1:end-23) '\cuda\bin\cws' '"'];
else
    disp('Platform not supported.');    
end

% check options
if nargin == 3
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
        opt.isverbose = 0;
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
    if length(opt.isverbose) ~= 1
        error('length of opt.isverbose must equal to 1!');
    end
else % use default parameters
    opt.priors = [0.1 0.1 100 5];
    opt.iter = [3 10 20];
    opt.mu = [0.1 100];
    opt.tol = 0.05;
    opt.L = min((2.^ceil(log2(size(I0))) - size(I0)) / 2, 256); % suit to [1024 1024]
    opt.isverbose = 0;
end

if isfield(opt,'size')
    opt.size = max(0, 2*round(opt.size/2));
    issize = [' -s ' num2str(opt.size(1))  ' -s ' num2str(opt.size(2))];
else % default the same size as inputs
    issize = [' -s ' num2str(size(I0,1))  ' -s ' num2str(size(I0,2))];
end

if abs(sign(opt.isverbose))
    isverbose_str = ' --verbose';
else
    isverbose_str = ' ';
end

% write input data to disk
J = cat(3, I0, I);
if ~isa(J,'uint8')
    J = uint8(255 * J / max(J(:)));
end
I_path = {['I1_' char(datetime('today')) '.png']; ...
          ['I2_' char(datetime('today')) '.png']};
arrayfun(@(i) imwrite(J(:,:,i), I_path{i}), 1:2);

% output name
out_name = ['out_tmp_' char(datetime('today'))  '.flo'];

% cat solver info
cws_run = [cws_path ...
    ' -p ' num2str(opt.priors(1)) ' -p ' num2str(opt.priors(2)) ...
    ' -p ' num2str(opt.priors(3)) ' -p ' num2str(opt.priors(4)) ...
    ' -i ' num2str(opt.iter(1))   ' -i ' num2str(opt.iter(2))   ...
    ' -i ' num2str(opt.iter(3))                                 ...
    ' -m ' num2str(opt.mu(1)) ' -m ' num2str(opt.mu(2))         ...
    ' -t ' num2str(opt.tol) isverbose_str issize                ...
    ' -l ' num2str(opt.L(1)) ' -l ' num2str(opt.L(2))...
    ' -o ' out_name ' -f "' I_path{1} '" -f "' I_path{2} '"'];

% run GPU solver
system(cws_run);

% read and load the results
test = readFlowFile(out_name);
A   = test(:,:,1);
phi = test(:,:,2);

% remove output file
if ismac || isunix
    system(['rm ' out_name]);
    arrayfun(@(i) system(['rm ' I_path{i}]), 1:numel(I_path));
elseif ispc
    system(['del ' out_name]);
    arrayfun(@(i) system(['del ' I_path{i}]), 1:numel(I_path));
else
    disp('Platform not supported.');
end

end


function img = readFlowFile(filename)

% readFlowFile read a flow file FILENAME into 2-band image IMG

%   According to the c++ source code of Daniel Scharstein
%   Contact: schar@middlebury.edu

%   Author: Deqing Sun, Department of Computer Science, Brown University
%   Contact: dqsun@cs.brown.edu
%   $Date: 2007-10-31 16:45:40 (Wed, 31 Oct 2006) $

% Copyright 2007, Deqing Sun.
%
%                         All Rights Reserved
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose other than its incorporation into a
% commercial product is hereby granted without fee, provided that the
% above copyright notice appear in all copies and that both that
% copyright notice and this permission notice appear in supporting
% documentation, and that the name of the author and Brown University not be used in
% advertising or publicity pertaining to distribution of the software
% without specific, written prior permission.
%
% THE AUTHOR AND BROWN UNIVERSITY DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
% INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
% PARTICULAR PURPOSE.  IN NO EVENT SHALL THE AUTHOR OR BROWN UNIVERSITY BE LIABLE FOR
% ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

TAG_FLOAT = 202021.25;  % check for this when READING the file

% sanity check
if isempty(filename) == 1
    error('readFlowFile: empty filename');
end

idx = strfind(filename, '.');
idx = idx(end);

if length(filename(idx:end)) == 1
    error('readFlowFile: extension required in filename %s', filename);
end

if strcmp(filename(idx:end), '.flo') ~= 1
    error('readFlowFile: filename %s should have extension ''.flo''', filename);
end

fid = fopen(filename, 'r');
if (fid < 0)
    error('readFlowFile: could not open %s', filename);
end

tag    = fread(fid, 1, 'float32');
width  = fread(fid, 1, 'int32');
height = fread(fid, 1, 'int32');

% sanity check

if (tag ~= TAG_FLOAT)
    error('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)', filename);
end

if (width < 1 || width > 99999)
    error('readFlowFile(%s): illegal width %d', filename, width);
end

if (height < 1 || height > 99999)
    error('readFlowFile(%s): illegal height %d', filename, height);
end

nBands = 2;

% arrange into matrix form
tmp = fread(fid, inf, 'float32');
tmp = reshape(tmp, [width*nBands, height]);
tmp = tmp';
img(:,:,1) = tmp(:, (1:width)*nBands-1);
img(:,:,2) = tmp(:, (1:width)*nBands);

fclose(fid);

end

