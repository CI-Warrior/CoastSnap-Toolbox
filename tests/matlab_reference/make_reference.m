% Generates tests/data/matlab_reference.mat from the original MATLAB code in
% rectifyCode/. Runs in MATLAB or GNU Octave (with the optim and statistics
% packages for nlinfit/tinv):
%
%   octave --no-gui -q tests/matlab_reference/make_reference.m
%
% Every input is deterministic so the Python tests can reproduce them.

here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..');
addpath(fullfile(root, 'rectifyCode'));
if exist('OCTAVE_VERSION', 'builtin')
    pkg load optim
    pkg load statistics
end
global globs

%% A. Distortion round trip with a real calibration (DJI Phantom 3, 4000x3000)
lcpA = makeLCPP3('Aerielle', 4000, 3000);
[gx, gy] = meshgrid(linspace(-1.2, 1.15, 9), linspace(-1.05, 1.0, 7));
A.u = gx(:)*lcpA.fx + lcpA.c0U;
A.v = gy(:)*lcpA.fy + lcpA.c0V;
[A.ud, A.vd] = distortCaltech(A.u, A.v, lcpA);
[A.uu, A.vu] = undistortCaltech(A.ud, A.vd, lcpA);
A.lcp = lcpA;

%% B. Projection with the CoastSnap profile and with the distorted profile
lcpB = makeLCPP3('CoastSnap', 1600, 1200);
B.beta = [2.5 -1.0 32.0 deg2rad([105 78 -1.5])];
[px, py] = meshgrid(linspace(80, 520, 6), linspace(-260, 220, 5));
% camera looks roughly east-south-east; put the points in front of it
B.xyz = [px(:) py(:) linspace(-0.5, 3, numel(px))'];
uv = findUVnDOF(B.beta, B.xyz, struct('lcp', lcpB));
B.uv = reshape(uv, [], 2);
B.lcp = lcpB;
B.uvA = reshape(findUVnDOF(B.beta, B.xyz, struct('lcp', lcpA)), [], 2);
B.xyz6 = findXYZ6dof(B.uv(:,1), B.uv(:,2), 0.7, B.beta, lcpB);
B.xyz6A = findXYZ6dof(B.uvA(:,1), B.uvA(:,2), 0.7, B.beta, lcpA);
B.P = lcpBeta2P(lcpB, B.beta);
B.m = P2m(B.P);
B.R = angles2R(0.3, 1.2, -0.05);

%% C. GCP solve with the FOV sweep, as in CSPGrectifyImage
NU = 1600; NV = 1200;
lcpC = makeLCPP3('CoastSnap', NU, NV);
fx_true = 0.5*NU/tan(62*pi/360);
lcpT = lcpC; lcpT.fx = fx_true; lcpT.fy = fx_true;
C.beta_true = [0 0 25 deg2rad([95 80 1.0])];
C.xyz = [ 150 -120 1.2;  220  -40 2.5;  300   60 0.8;  180  130 4.0;
          400 -200 0.5;  500   10 1.5;  260 -250 3.0;  350  200 2.0];
uvT = reshape(findUVnDOF(C.beta_true, C.xyz, struct('lcp', lcpT)), [], 2);
noise = [ 0.8 -1.1; -0.6 0.4; 1.3 0.2; -0.2 -0.9; 0.5 1.0; -1.0 -0.3; 0.1 0.7; 0.9 -0.5];
C.uv = uvT + noise;
C.fov_lims = [50 75];
C.beta0 = [0 0 25 deg2rad([100 75 0])];
knownFlags = [1 1 1 0 0 0];
globs.lcp = lcpC; globs.knownFlags = knownFlags; globs.knowns = C.beta0(logical(knownFlags));
b0 = C.beta0(~knownFlags);
fx_max = 0.5*NU/tan(C.fov_lims(1)*pi/360);
fx_min = 0.5*NU/tan(C.fov_lims(2)*pi/360);
fx_min = interp1([5:5:500000],[5:5:500000],fx_min,'nearest');
fx_max = interp1([5:5:500000],[5:5:500000],fx_max,'nearest');
C.fx = fx_min:5:fx_max;
opts = statset('TolFun', 1e-12, 'TolX', 1e-12);
C.mse_all = zeros(size(C.fx));
for i = 1:length(C.fx)
    g = globs; g.lcp.fx = C.fx(i); g.lcp.fy = C.fx(i);
    findUVnDOF(b0, C.xyz, g);   % sets the global that nlinfit's 2-argument calls use
    [b, R] = nlinfit(C.xyz, [C.uv(:,1); C.uv(:,2)], @(b, x) findUVnDOF(b, x), b0, opts);
    % MATLAB's nlinfit MSE is SSR/(numel(Y) - numel(beta)); Octave's optim
    % package divides by numel(X) - numel(beta) instead, so compute it here.
    C.mse_all(i) = (R'*R)/(numel(R) - numel(b));
end
[~, Imin] = min(C.mse_all);
globs.lcp.fx = C.fx(Imin); globs.lcp.fy = C.fx(Imin);
findUVnDOF(b0, C.xyz, globs);
[beta, R, J, CovB, mse] = nlinfit(C.xyz, [C.uv(:,1); C.uv(:,2)], @(b, x) findUVnDOF(b, x), b0, opts);
dof = numel(R) - numel(beta);
half = tinv(0.975, dof) * sqrt(diag(CovB))';
C.betas = [globs.knowns beta(:)'];
C.ci = [zeros(2,3) [beta(:)' - half; beta(:)' + half]];
C.mse = (R'*R)/dof;
C.fx_best = C.fx(Imin);

%% D. Rectification products (buildRectProducts + makeFinalImages)
NU = 320; NV = 240;
lcpD = makeLCPP3('CoastSnap', NU, NV);
[uu, vv] = meshgrid(1:NU, 1:NV);
I1 = zeros(NV, NU, 3);
I1(:,:,1) = mod(uu*7 + vv*3, 256);
I1(:,:,2) = mod(uu.*vv, 251);
I1(:,:,3) = 128 + 100*sin(uu/9).*cos(vv/13);
I1 = round(I1);
I2 = round(mod(I1*1.7 + 40, 256));
D.I1 = uint8(I1); D.I2 = uint8(I2);
D.beta1 = [0 0 20 deg2rad([90 75 0.5])];
D.beta2 = [0 0 20 deg2rad([90.4 75.3 0.2])];
D.rectxy = [40 2 300 -120 2 120];
D.z = 0.6;
images.xy = D.rectxy; images.z = D.z;
images = buildRectProducts(1, images, D.I1, D.beta1, struct('lcp', lcpD));
f1 = makeFinalImages(images);
D.timex1 = f1.timex; D.N1 = f1.N; D.x = f1.x; D.y = f1.y;
images = buildRectProducts(2, images, D.I2, D.beta2, struct('lcp', lcpD));
f2 = makeFinalImages(images);
D.timex2 = f2.timex; D.bright2 = f2.bright; D.dark2 = f2.dark; D.N2 = f2.N;
% same rectification with the distorted Phantom calibration on a 4000x3000 frame
lcpE = lcpA;
[uu, vv] = meshgrid(1:4000, 1:3000);
IE = zeros(3000, 4000, 3);
IE(:,:,1) = mod(uu + 2*vv, 256); IE(:,:,2) = mod(3*uu - vv, 256); IE(:,:,3) = mod(uu.*vv, 256);
D.betaE = [0 0 60 deg2rad([0 60 0])];
D.rectxyE = [-80 4 80 30 4 150];
imagesE.xy = D.rectxyE; imagesE.z = 0;
imagesE = buildRectProducts(1, imagesE, IE, D.betaE, struct('lcp', lcpE));
fE = makeFinalImages(imagesE);
D.timexE = fE.timex; D.NE = fE.N;

save('-v7', fullfile(root, 'tests', 'data', 'matlab_reference.mat'), 'A', 'B', 'C', 'D');
disp('wrote tests/data/matlab_reference.mat')
