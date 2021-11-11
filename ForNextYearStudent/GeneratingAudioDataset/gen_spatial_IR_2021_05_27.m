function[hspatial,radius,max_radius_val] = gen_spatial_IR_2021_05_27(az_angle, el_angle, radius_ratio, room_dimensions, alpha)

c = 340;                    % Sound velocity (m/s)
fs = 16000;                 % Sample frequency (samples/s)
L = room_dimensions;  % Room dimensions [x y z] (m)
c_x = L(1)/2;
c_y = L(2)/2;
c_z = L(3)/2;

n = 4096;                   % Number of samples
mtype = 'omnidirectional';  % Type of microphone
order = -1;                 % -1 equals maximum reflection order!
dim = 3;                    % Room dimension
orientation = 0;            % Microphone orientation (rad)
hp_filter = 1;              % Enable high-pass filter

r = [c_x, c_y, c_z];
el_rad = el_angle*pi/180;
az_rad = az_angle*pi/180;
max_radius = ones(1,3)*max(room_dimensions);
if cos(el_rad) ~= 0
    if cos(az_rad) ~= 0
        max_radius(1) = abs((L(1) - c_x)/(cos(el_rad)*cos(az_rad)));
    end
    if sin(az_rad) ~= 0
        max_radius(2) = abs((L(2) - c_y)/(cos(el_rad)*sin(az_rad)));
    end
end
if sin(el_rad) ~= 0
    max_radius(3) = abs((L(3) - c_z)/(sin(el_rad)));
end

max_radius_val = min(max_radius);


radius = radius_ratio*max_radius_val;
[x_target, y_target, z_target] = sph2cart(az_rad,el_rad,radius);
target_pos = [c_x + x_target, c_y + y_target, c_z + z_target];              % src position [x y z] (m)
hspatial = rir_generator(c, fs, r, target_pos, L, alpha, n, mtype, order, dim, orientation, hp_filter).';


