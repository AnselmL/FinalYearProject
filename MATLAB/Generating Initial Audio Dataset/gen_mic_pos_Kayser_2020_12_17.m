function[sensor_pos] = gen_mic_pos_Kayser()
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