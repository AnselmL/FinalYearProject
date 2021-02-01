clear all;
addpath(genpath('MBSTOI'));
addpath(genpath('submodules'));
addpath(genpath('other_dependencies'));
addpath(genpath('RIR-Generator-master'));
addpath(genpath('ANF-Generator-master'));


%{
[x, fs_x] = audioread('anechoic_speech.wav');
y = rand(size(x,1),1);

[z,p,fs_o] = v_addnoise(x,fs_x,10,'dxopEkn',y,fs_x);
%}


base_dir = 'E:\FYP\MATLAB';
RIR_dir = 'E:\FYP\Room_IR_Database';
RIR_file_ext = '.wav';

out_base_dir = 'E:\FYP\ML_data_dir\1-fold';

az_target = -180:10:170;
%az_target_sm_rm = -180:12.5:170;
el_target = [-10,10];
%az_noise = [90];
%el_noise = [0];

%az_el_target = [0,0]; %this can also be matrix if multiple target azimuths and elevations are desired
%az_el_noise = [90,0]; %this can be matrix with column 1 for azimuth angle and column 2 for elevation angle
radii_target = [0.8,3];%row matrix of all desired target radii, specified in meters
radii_target_sm_rm = ones(1,1);
radii_target_sm_rm(1) = 0.8;
%radii_noise = [1,3,5]; %row matrix of all desired noise radii, specified in 
noise_name_array = {'NOISEX-92','babble';
                    'NOISEX-92','buccaneer1';
                    'NOISEX-92','buccaneer2';
                    'NOISEX-92','destroyerengine';
                    'NOISEX-92','destroyerops';
                    'NOISEX-92','f16';
                    'NOISEX-92','factory1';
                    'NOISEX-92','factory2';
                    'NOISEX-92','hfchannel';
                    'NOISEX-92','leopard';
                    'NOISEX-92','m109';
                    'NOISEX-92','machinegun';
                    'NOISEX-92','pink';
                    'NOISEX-92','volvo';
                    'NOISEX-92','white';
                    'RSG-10','SIGNAL001-20kHz';
                    'RSG-10','SIGNAL002-20kHz';
                    'RSG-10','SIGNAL003-20kHz';
                    'RSG-10','SIGNAL004-20kHz';
                    'RSG-10','SIGNAL005-20kHz';
                    'RSG-10','SIGNAL006-20kHz';
                    'RSG-10','SIGNAL007-20kHz';
                    'RSG-10','SIGNAL008-20kHz';
                    'RSG-10','SIGNAL009-20kHz';
                    'RSG-10','SIGNAL010-20kHz';
                    'RSG-10','SIGNAL011-20kHz';
                    'RSG-10','SIGNAL012-20kHz';
                    'RSG-10','SIGNAL013-20kHz';
                    'RSG-10','SIGNAL014-20kHz';
                    'RSG-10','SIGNAL015-20kHz';
                    'RSG-10','SIGNAL016-20kHz';
                    'RSG-10','SIGNAL017-20kHz';
                    'RSG-10','SIGNAL018-20kHz';
                    'RSG-10','SIGNAL019-20kHz';
                    'RSG-10','SIGNAL020-20kHz';
                    'RSG-10','SIGNAL021-20kHz';
                    'RSG-10','SIGNAL022-20kHz';
                    'RSG-10','SIGNAL023-20kHz';
                    'RSG-10','SIGNAL024-20kHz';
                    };
                
sensor_pos = gen_mic_pos_Kayser();
for i = 1:size(noise_name_array,1)
    noise_dir = strcat(base_dir,'\',noise_name_array{i,1},'\',noise_name_array{i,2},'.wav');
    [y,fs] = audioread(noise_dir);
    y = resample(y,16000,fs);
    fs = 16000;
    full_noise_array(i,:,:) = gen_isotropic_noise(y,fs,sensor_pos);
end
save('full_noise_array','full_noise_array');
full_noise_array = load('full_noise_array','full_noise_array').full_noise_array;
TIMIT_wav_dirs_train = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_train;
TIMIT_wav_dirs_val = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_val;
TIMIT_wav_dirs_test = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_test;

mode = ["train","val","test"];


SNR = -30:3:30;


%radii_total = unique(cat(2,radii_target,radii_noise));
%az_total = unique(cat(2,az_target,az_noise));
%el_total = unique(cat(2,el_target,el_noise));

dimensions = [2.5,2.5,3; 4,4,3; 6,6,3; 7.5,7.5,3;10,10,4; 12.5,12.5,4; 15,15,4;10,20,4; 20,20,5; 25,25,10;10,30,4; 30,30,15]; % room dimensions specified with [x,y,z] coordinates
alpha = [0.6,0.4,0.2]; %absorption coefficients
rt60 = [];


noise_index = ;
prev_used_noise_index = [];

prev_used_TIMIT_index = [];
SNR_index = ;
prev_used_SNR_index = [];
alpha_index = ;
prev_used_alpha_index = [];
room_size_index = ;
prev_used_room_size_index = [];
radius_index = ;
prev_used_radius_index = [];

TIMIT_index = nearest(rand*size(TIMIT_wav_dirs,1));
if ismember(TIMIT_index,prev_used_TIMIT_index)
    while ~ismbember(TIMIT_index,prev_used_TIMIT_index)
        TIMIT_index = nearest(rand*size(TIMIT_wav_dirs,1));
    end
end
prev_used_TIMIT_index = [TIMIT_index,prev_used_TIMIT_index];

anechoic_speech_dir = TIMIT_wav_dirs{TIMIT_index,1};

[anechoic_speech, fs_speech] = audioread(anechoic_speech_dir);


->fetch impulse response
%kayser

noise_index = nearest(rand * size(noise_name_array,1));
if ismember(noise_index,prev_used_index)
    while ~ismbember(noise_index,prev_used_index)
        noise_index = nearest(rand);
    end
end
prev_used_index = [prev_used_index, noise_index];
index_check = ismember(index_test,prev_used_index);
if all(index_check)
    prev_used_index = [];
end
