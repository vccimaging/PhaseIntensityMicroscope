function phi = tilt_removal(phi)

phi = phi - min(phi(:));
[x,y] = meshgrid(1:size(phi,2),1:size(phi,1));
[~,~,~,S] = affine_fit(x,y,phi);
phi = phi - S;
phi = phi - min(phi(:));

end


function [n,V,p,S] = affine_fit(x,y,z)
    %Computes the plane that fits best (lest square of the normal distance
    %to the plane) a set of sample points.
    %INPUTS:
    %x: 2D x coordinates
    %y: 2D y coordinates
    %z: input 2D array to be fit
    % (org) X: a N by 3 matrix where each line is a sample point
    %
    %OUTPUTS:
    %
    %n : a unit (column) vector normal to the plane
    %V : a 3 by 2 matrix. The columns of V form an orthonormal basis of the
    %plane
    %p : a point belonging to the plane
    %S : the fitted affine plane
    %
    %NB: this code actually works in any dimension (2,3,4,...)
    %Author: Adrien Leygue
    %Date: August 30 2013
    
    X = [x(:) y(:) z(:)];
    
    %the mean of the samples belongs to the plane
    p = mean(X,1);
    
    %The samples are reduced:
    R = bsxfun(@minus,X,p);
    %Computation of the principal directions if the samples cloud
    [V,~] = eig(R'*R);
    %Extract the output from the eigenvectors
    n = V(:,1);
    V = V(:,2:end);
    
    % the fitted plane
    S = - (n(1)/n(3)*x + n(2)/n(3)*y - dot(n,p)/n(3));
end
