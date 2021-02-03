function[hspatial] = gen_spatial_IR(az_angle, el_angle, radius, room_dimensions, alpha,rt60)



mic_radius = 0.09;%obj.sphereRadius; %0.082;     % radius on which microphones lie [metres]
mic_spacing = 0.015;   % distance between front and back microphones [metres]
%r_d = 3;

angle_offset = deg2rad(7.5); % line bisecting microphones is offset back from a line through the sphere
angle_spacing = asin(mic_spacing/2 /mic_radius);

az = [pi/2 + angle_offset - angle_spacing; ...%front left
    -(pi/2 + angle_offset) + angle_spacing; ...%front right
    pi/2 + angle_offset + angle_spacing; ...%rear left
    -(pi/2 + angle_offset) - angle_spacing];   %rear right
inc = pi/2 * ones(4,1);

sensor_pos = mic_radius * [cos(az).*sin(inc), sin(az).*sin(inc), cos(inc)]; % [x,y,z] offsets of sensors
rot_mat = v_rotation(sensor_pos(3,:).',sensor_pos(1,:).',pi/2);
sensor_pos = (rot_mat*sensor_pos.').';
c = 340;                    % Sound velocity (m/s)
fs = 16000;                 % Sample frequency (samples/s)
%r = [4.5 4.5 2 ;5.5 4.5 2; 4.5 5.5 2;5.5 5.5 2];% Receiver positions [x_1 y_1 z_1 ; x_2 y_2 z_2] (m)
L = room_dimensions;  % Room dimensions [x y z] (m)
c_x = L(1)/2;
c_y = L(2)/2;
c_z = L(3)/2;
V = L(1)*L(2)*L(3);
S = 2*(L(1)*L(3)+L(2)*L(3)+L(1)*L(2));
alpha = 0.5;
if isempty(rt60)
    beta = alpha*24*V*log(10.0)/(c*S); %using alpha = 0.5
else
    beta = rt60; % Reverberation time (s)
end
   
n = 4096;                   % Number of samples
mtype = 'omnidirectional';  % Type of microphone
order = -1;                 % -1 equals maximum reflection order!
dim = 3;                    % Room dimension
orientation = 0;            % Microphone orientation (rad)
hp_filter = 1;              % Enable high-pass filter

r = [c_x + sensor_pos(1,1), c_y + sensor_pos(1,2), c_z + sensor_pos(1,3) ; c_x + sensor_pos(2,1), c_y + sensor_pos(2,2), c_z + sensor_pos(2,3);c_x + sensor_pos(3,1), c_y + sensor_pos(3,2), c_z + sensor_pos(3,3) ;c_x + sensor_pos(4,1), c_y + sensor_pos(4,2), c_z + sensor_pos(4,3)];


[x_target, y_target, z_target] = sph2cart(az_angle*pi/180,el_angle*pi/180,radius);

target_pos = [c_x + x_target, c_y + y_target, c_z + z_target];              % src position [x y z] (m)
hspatial = rir_generator(c, fs, r, target_pos, L, beta, n, mtype, order, dim, orientation, hp_filter);

