% Generates tests/data/utils_reference.mat from the MATLAB file-naming, time,
% UTM and shoreline-shift code. Runs in MATLAB or GNU Octave:
%
%   octave --no-gui -q tests/matlab_reference/make_utils_reference.m
%
% CSPargusFilename looks the site's time zone up with CSPreadSiteDB, which
% reads the Excel DB; a stub returning the global TZ stands in for it.

here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..');
stub = fullfile(tempdir, 'coastsnap_stub');
if ~exist(stub, 'dir'), mkdir(stub); end
fid = fopen(fullfile(stub, 'CSPreadSiteDB.m'), 'w');
fprintf(fid, 'function out = CSPreadSiteDB(site)\nglobal TZ\nout.timezone = TZ;\n');
fclose(fid);
if ~exist('polyxpoly')
    % Mapping Toolbox function; Octave's mapping package lacks it. Segment
    % intersections in order along the first line, as MATLAB returns them.
    fid = fopen(fullfile(stub, 'polyxpoly.m'), 'w');
    fprintf(fid, ['function [xi, yi] = polyxpoly(x1, y1, x2, y2)\n' ...
        'xi = []; yi = [];\n' ...
        'for a = 1:numel(x1)-1\n for b = 1:numel(x2)-1\n' ...
        '  p = [x1(a) y1(a)]; r = [x1(a+1) y1(a+1)] - p;\n' ...
        '  q = [x2(b) y2(b)]; s = [x2(b+1) y2(b+1)] - q;\n' ...
        '  d = r(1)*s(2) - r(2)*s(1); if d == 0, continue; end\n' ...
        '  t = ((q(1)-p(1))*s(2) - (q(2)-p(2))*s(1))/d; u = ((q(1)-p(1))*r(2) - (q(2)-p(2))*r(1))/d;\n' ...
        '  if t >= 0 && t <= 1 && u >= 0 && u <= 1, xi(end+1,1) = p(1)+t*r(1); yi(end+1,1) = p(2)+t*r(2); end\n' ...
        ' end\nend\n']);
    fclose(fid);
end
addpath(fullfile(root, 'rectifyCode'), fullfile(root, 'tools'), fullfile(root, 'GUI'), root);
addpath(stub);  % in front of the real CSPreadSiteDB
cd(stub);       % the current folder would win otherwise
global TZ

%% Argus file names
N.epochs = [1528322400 1131267000 9000 1700000000.7 1600000000 1609459199];
N.tz_names = {'AEST', 'ACST', 'GMT', 'PST'};
N.tz_offsets = [10 9.5 0 -8];
N.users = {'Mitch', ''};
N.names = {};
N.short = {};
k = 0;
for i = 1:numel(N.epochs)
    for j = 1:numel(N.tz_names)
        TZ.name = N.tz_names{j};
        TZ.gmt_offset = N.tz_offsets(j);
        for u = 1:numel(N.users)
            k = k + 1;
            N.names{k} = CSPargusFilename(N.epochs(i), 'manly', -1, 'snap', N.users{u}, 'jpg');
        end
    end
end
N.cam2 = CSPargusFilename(1528322400, 'manly', 2, 'timex', 'Mitch', 'jpg');

%% Times
T.epochs = N.epochs;
T.datenums = epoch2Matlab(T.epochs);
T.back = matlab2Epoch(T.datenums(:));
T.julian = arrayfun(@(d) matlab2Julian(d), T.datenums);
T.argusday = {};
for i = 1:numel(T.epochs)
    T.argusday{i} = argusDay(floor(T.epochs(i)));
end
T.local = CSPepoch2LocalMatlab(T.epochs, 10);

%% UTM
U.x = [342000 345678.9 500000 712345];
U.y = [6265000 6234567.8 6000000 3456789];
U.zones = {'56 H', '56 H', '56 J', '33 T'};
for i = 1:numel(U.x)
    [U.lat(i), U.lon(i)] = utm2deg(U.x(i), U.y(i), U.zones{i});
end

%% Shoreline shift along transects (CSPGshiftSLxshore). Transects run east
%% from a landward start, so MATLAB's +x shift is seaward.
ys = 5:10:195;
S.transects.x = [5*ones(size(ys)); 95*ones(size(ys))];
S.transects.y = [ys; ys + 3];
yy = (0:2:200)';
S.sl.xyz = [40 + 0.1*yy + 2*sin(yy/15), yy, 0.4*ones(size(yy))];
S.shift = 7.5;
S.new = CSPGshiftSLxshore(S.sl, S.transects, S.shift, 1.2);

out = fullfile(root, 'tests', 'data', 'utils_reference.mat');
save('-v7', out, 'N', 'T', 'U', 'S');
printf('wrote %s\n', out);
