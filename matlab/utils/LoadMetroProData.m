%
% function [ dat, xl, yl ] = LoadMetroProData( filename )
%
% Read MetroPro binary data file and returns in dat. If failed, dat is empty.
%
% x and y coordinates of data points are calculaed as (see NB below)
%     xl = dxy * ( -(size(dat,2)-1)/2 : (size(dat,2)-1)/2 );
%     yl = dxy * ( -(size(dat,1)-1)/2 : (size(dat,1)-1)/2 );
%
% dat(iy,jx) is the value at the location (x,y) = ( xl(jx), yl(iy) )
%   which is the convention of matlab
%
% The following commands will show data
%   plot( xl, dat(floor(end/2), :), yl, dat(:, floor(end/2) )
%   mesh( xl, yl, dat*1e9 )
%
% NB) aboslute values of coordinate is not useful,
% because the mesurement origin (0,0) is located at the top-left corner
% and not at the center of the mirror.
%
function [ dat, xl, yl ] = LoadMetroProData( filename )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% March 20, 2014
%   Zernike term/tilt handling has been removed, and this function works without other functions
%
% September 17, 2013
%   Third output argument, xl, added
%
% September 9, 2013 HY
%   zernike term removal added
%
% August 30, 2013  HY
%   1) piston/tilt removal is added
%   2) matrxi transversed to follow matlab convention
%
%   MetroPro format stores data row major, starting from top-left
%   I.e., in units of spacing dxy, data at the following locations are stored in sequence
%   (x,y) = (1,1), (2,1), (3,1)...
%   When this series of data is read in into matlab array dat, the data at (i,j) is loaded at dat(i,j)
%   To use the matlab convention, i.e., x axis horizontal or second column index, matrix is transvered.
%   Now dat(iy, jx) is the value at phyical location (jx, iy)*dxy.
%
%   When using a nominal x-y convention, positive directions of x and y axises are right and up.
%   The data need to be revered in the y-direction. Because ...
%   The first data at top left is at (x,y) = (1,1) and the last data at bot-right is at (N,N).
%   For x axis, this is the correct ordering (1 to N from left to right),
%   but, for y, it is opposite (1 to N from top to bottom).
%   To change the y ordering, data(iy, jx) needs to be data(N-iy+1, jx) or flipud.
%   This makes the phyical location of y for dat(1,j) is below of dat(2,j).
%
% August 19, 2013  Hiro Yamamoto
%   Based on MetroPro Reference Guide,
%   Section 12 "Data Format and Conversion" which covers
%   header format 2 and 3 and phase_res format 0, 1 and 2.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% try to open the file
fid = fopen( filename, 'r', 'b' );
if fid == -1
    error('File ''%s'' does not exist', filename);
end;

% read the header information
hData = readerHeader( fid );
if hData.hFormat < 0
    fclose(fid);
    error('Format unknown');
end;

% read the phasemap data
% skip the header and intensity data
fseek( fid, hData.hSize + hData.IntNBytes, -1 );
[dat, count] = fread( fid, hData.XN*hData.YN, 'int32' );
fclose( fid );

if count ~= hData.XN*hData.YN
    error('data could not fully read');
end

% mark unmeasured data as NaN
dat( dat >= hData.invalid ) = NaN;
% scale data to unit of meter
dat = dat * hData.convFactor;
% reshape data to XN x YN matrux
dat = reshape( dat, hData.XN, hData.YN );
% transpose to make the matrix (NY, NX)
dat = dat';
% change the y-axis diretion
dat = flipud( dat );

% auxiliary data to return
dxy = hData.CameraRes;

if nargout >= 2
    xl = dxy * ( -(size(dat,2)-1)/2 : (size(dat,2)-1)/2 );
end
if nargout >= 3
    yl = dxy * ( -(size(dat,1)-1)/2 : (size(dat,1)-1)/2 );
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read header information, and hData.hFormat = -1 when failed.
% information of the header segment is from MetroPro Reference Guide
% The manual covers format 1, 2 and 3, and this function fails if the data has unknown format.

function hData = readerHeader( fid )

% first, check the format information to make sure this is MetroPro binary data
hData.hFormat = -1;
[magicNum, count] = fread( fid, 1, 'uint32' );
if count == 0
    return;
end;

[hData.hFormat, count] = readI16( fid );
if count == 0
    return;
end;

[hData.hSize, count] = readI32( fid );
if count == 0
    return;
end;

% check if the magic string and format are known ones
if (hData.hFormat >= 1) && (hData.hFormat<=3) && (magicNum-hData.hFormat == hex2dec('881B036E'))
    %    sprintf('MetroPro format %d with header size %d', hData.hFormat, hData.hSize )
else
%    sprintf('====> warning format unknown : %d\n', hData.hFormat);
     hData.hFormat = -1;
     return;
end

% read necessary data
hData.invalid = hex2dec('7FFFFFF8');

% intensitity data, which we will skip over
hData.IntNBytes = readI32( fid, 61-1 );

% top-left coordinate, which are useless
hData.X0 = readI16( fid, 65-1 );
hData.Y0 = readI16( fid, 67-1 );

% number of data points along x and y
hData.XN = readI16( fid, 69-1 );
hData.YN = readI16( fid, 71-1 );

% total data, 4 * XN * YN
hData.PhaNBytes = readI32( fid, 73-1 );

% scale factor is determined by phase resolution tag
phaseResTag = readI16( fid, 219-1 );
switch phaseResTag
    case 0,
        phaseResVal = 4096;
        
    case 1,
        phaseResVal = 32768;
        
    case 2,
        phaseResVal = 131072;
        
    otherwise
        phaseResVal = 0;
end

hData.waveLength = readReal( fid, 169-1 );
% Eq. in p12-6 in MetroPro Reference Guide
hData.convFactor = readReal( fid, 165-1 ) * readReal( fid, 177-1 ) * hData.waveLength / phaseResVal;

% bin size of each measurement
hData.CameraRes = readReal( fid, 185-1 );

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% utility to read data, which are stored in big-endian format
function [val, count] = readI16( fid, offset )
if nargin == 2
    fseek( fid, offset, -1 );
end
[val, count] = fread( fid, 1, 'int16' );
return

function [val, count] = readI32( fid, offset )
if nargin == 2
    fseek( fid, offset, -1 );
end
[val, count] = fread( fid, 1, 'int32' );
return

function [val, count] = readReal( fid, offset )
if nargin == 2
    fseek( fid, offset, -1 );
end
[val, count] = fread( fid, 1, 'float' );
return