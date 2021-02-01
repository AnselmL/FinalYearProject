clear all;
addpath(genpath('MBSTOI'));
addpath(genpath('submodules'));
addpath(genpath('other_dependencies'));
addpath(genpath('RIR-Generator-master'));
addpath(genpath('ANF-Generator-master'));


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
TIMIT_wav_dirs = load('TIMIT_wav_dirs').TIMIT_wav_dirs;




%radii_total = unique(cat(2,radii_target,radii_noise));
%az_total = unique(cat(2,az_target,az_noise));
%el_total = unique(cat(2,el_target,el_noise));

dimensions = [2.5,2.5,3; 4,4,3; 6,6,3; 7.5,7.5,3;10,10,4; 12.5,12.5,4; 15,15,4;10,20,4; 20,20,5; 25,25,10;10,30,4; 30,30,15]; % room dimensions specified with [x,y,z] coordinates
alpha = [0.5,
rt60 = [];






%if simulated_IR
%{
for i = 1:size(dimensions,1)
    room_dim = dimensions(i,:);
    %{
    if graphing == 1
        [h_target, h_noise,receiver_pos,target_pos,noise_pos] = gen_spatial_IR_vectorised(az_target, el_target, az_noise, el_noise, radii_target, radii_noise, room_dim, rt60);
    elseif graphing == 0
    %}
    room_folder = strcat('x_',num2str(dimensions(i,1)),'_y_',num2str(dimensions(i,2)),'_z_',num2str(dimensions(i,3)));
    RIR_room_folder = strcat(RIR_dir,'\',room_folder);
    if ~exist(RIR_room_folder, 'dir')
        mkdir(RIR_room_folder)
    end
    if i == 1
        for j = 1:size(radii_target_sm_rm,2)
            for k = 1:size(az_target_sm_rm,2)
                for l = 1:size(el_target,2)
                    [h] = gen_spatial_IR(az_target_sm_rm(k), el_target(l), radii_target_sm_rm(j), room_dim, rt60);
                    if size(h,1) < size(h,2)
                        h = permute(h,[2,1]);
                    end
                    file_name = strcat('r_',num2str(radii_target(j)),'_az_',num2str(az_target_sm_rm(k)),'_el_',num2str(el_target(l)));
                    file_name_dir = strcat(RIR_room_folder,'\',file_name,RIR_file_ext);
                    v_writewav(h,16000,file_name_dir,[],[],[],[]);
                end
            end
        end
    else
        for j = 1:size(radii_target,2)
            for k = 1:size(az_target,2)
                for l = 1:size(el_target,2)
                    [h] = gen_spatial_IR(az_target(k), el_target(l), radii_target(j), room_dim, rt60);
                    if size(h,1) < size(h,2)
                        h = permute(h,[2,1]);
                    end
                    file_name = strcat('r_',num2str(radii_target(j)),'_az_',num2str(az_target(k)),'_el_',num2str(el_target(l)));
                    file_name_dir = strcat(RIR_room_folder,'\',file_name,RIR_file_ext);
                    v_writewav(h,16000,file_name_dir,[],[],[],[]);
                end
            end
        end
    end

end
%}
%end
disp('impulse responses have been generated');
count = 1;
%%%string for kayser directory
%%array of kayser names
%run this as you would earlier
index_test = 1:size(noise_name_array,1);
prev_used_index = [];
for i = 1:size(dimensions,1)
    disp(i);
    out_1st_lev_dir = strcat(out_base_dir,'\','x_',num2str(dimensions(i,1)),'_y_',num2str(dimensions(i,2)),'_z_',num2str(dimensions(i,3)));
    if ~exist(out_1st_lev_dir, 'dir')
        mkdir(out_1st_lev_dir)
    end
    if i == 1  %small room
        radii = radii_target_sm_rm;
        az = az_target_sm_rm;
    else
        radii = radii_target;
        az = az_target;
    end
    for j = 1:size(az,2)
        for k = 1:size(el_target,2)
            for l = 1:size(radii,2)
                %for m = 1:size(full_noise_array,1)
                anechoic_speech_dir = TIMIT_wav_dirs{count,1};
                [anechoic_speech, fs_speech] = audioread(anechoic_speech_dir);
                noise_index = nearest(rand);
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
                if size(prev_used_index,2) > size(noise_name_array,1)
                    error('size of previously_used_index is greater than all possible indeces')
                end

                room_folder = strcat('x_',num2str(dimensions(i,1)),'_y_',num2str(dimensions(i,2)),'_z_',num2str(dimensions(i,3)));
                RIR_room_folder = strcat(RIR_dir,'\',room_folder);

                rir_file_name = strcat('r_',num2str(radii(l)),'_az_',num2str(az(j)),'_el_',num2str(el_target(k)));
                rir_dir = strcat(RIR_room_folder,'\',rir_file_name,RIR_file_ext);
                [rir, rir_fs] = audioread(rir_dir);
                if rir_fs ~= fs_speech
                    rir = resample(rir,fs_speech,rir_fs);
                end
                rev_speech_4ch = fftfilt(rir,anechoic_speech);
                noise_4ch = squeeze(full_noise_array(noise_index,:,:));
                noise_4ch = noise_4ch(1:size(rev_speech_4ch,1),:);

                rev_speech_2ch = [(rev_speech_4ch(:,1) + rev_speech_4ch(:,3))/2,(rev_speech_4ch(:,2) + rev_speech_4ch(:,4))/2];
                %rev_speech_2ch(:,1) = (rev_speech_4ch(:,1) + rev_speech_4ch(:,3))/2;
                %rev_speech_2ch(:,2) = (rev_speech_4ch(:,2) + rev_speech_4ch(:,4))/2;

                noise_2ch = [(noise_4ch(:,1) + noise_4ch(:,3))/2,(noise_4ch(:,2) + noise_4ch(:,4))/2];
                %noise_2ch(:,1) = (noise_4ch(:,1) + noise_4ch(:,3))/2;
                %noise_2ch(:,2) = (noise_4ch(:,2) + noise_4ch(:,4))/2;
                out_2nd_lev_dir = strcat(out_1st_lev_dir,'\',rir_file_name);
                out_3rd_lev_dir = strcat(out_2nd_lev_dir,'\',noise_name_array(m,1),'_',noise_name_array(m,2));
                out_3rd_lev_dir = out_3rd_lev_dir{1};
                if ~exist(out_2nd_lev_dir, 'dir')
                    mkdir(out_2nd_lev_dir)
                end
                count_snr = 1;
                for snrs = -30:3:30
                    s_db=v_activlev(rev_speech_2ch,16000,'d');  % speech level in dB
                    mod_rev_speech = 10^((-(s_db))/20)*rev_speech_2ch;
                    s_db=v_activlev(mod_rev_speech,16000,'d');  % speech level in dB
                    mod_rev_speech = 10^((-(s_db))/20)*mod_rev_speech;
                    s_db = v_activlev(mod_rev_speech,16000,'d');  % speech level in dB
                    n_db = mean(pow2db(sum(noise_2ch.^2)/size(noise_2ch,1)));
                    mixed_out = mod_rev_speech + 10^(-(snrs + n_db)/20)*noise_2ch;


                    %for speech_type = 1:2
                    %if speech_type == 1
                    xl = anechoic_speech;
                    xr = anechoic_speech;
                    %{
                    elseif speech_type == 2
                        xl = mod_rev_speech(:,1);
                        xr = mod_rev_speech(:,2);
                    end
                    %}

                    yl = mixed_out(:,1);
                    yr = mixed_out(:,2);

                    [mbstoi_full, mbstoi_intermediate] = mbstoi(xl,xr,yl,yr,fs_speech);
                    %directory of where to save mat file with
                    %mbstoi

                    out_snr_lev_dir = strcat(out_3rd_lev_dir,'\',num2str(count_snr));
                    if ~exist(out_snr_lev_dir, 'dir')
                        mkdir(out_snr_lev_dir)
                    end
                    save(strcat(out_snr_lev_dir,'\','mbstoi_label'),'mbstoi_full','mbstoi_intermediate');
                    anechoic_write_dir = strcat(out_snr_lev_dir,'\','anechoic_speech.wav');
                    v_writewav(anechoic_speech,16000,anechoic_write_dir,[],[],[],[]);
                    mixed_write_dir = strcat(out_snr_lev_dir,'\','mixed.wav');
                    v_writewav(mixed_out,16000,mixed_write_dir,[],[],[],[]);
                    %log file
                    fileID = fopen(strcat(out_snr_lev_dir,'\','log.txt'),'w');
                    fprintf(fileID,'mbstoi_full contains mbstoi value of entire sentence \n');
                    fprintf(fileID,'mbstoi_intermediate contains intermediate mbstoi values across sentence \n');
                    fprintf(fileID,'for both of these the clean reference speech was anechoic \n');
                    %fprintf(fileID,'For both of these index 1 and 2 are using anechoic and reverberant speech as clean respectively \n');
                    fprintf(fileID,'\n\n');
                    fprintf(fileID,strcat(strrep(strcat('directory of TIMIT file used: ',TIMIT_wav_dirs{count,1}),'\','/'),' \n'));
                    fprintf(fileID,'SNR: %.1f',snrs);
                    fprintf(fileID,'\n\n');
                    fprintf(fileID,'Impulse Response Parameters: \n');
                    fprintf(fileID,'Simulated \n');
                    fprintf(fileID,'room dimensions [%.1f,%.1f,%.1f] [x,y,z] \n',dimensions(i,1),dimensions(i,2),dimensions(i,3));
                    fprintf(fileID,'radius of target: %.2f \n',radii(l));
                    fprintf(fileID,'azimuth angle of target: %.2f \n',az(j));
                    fprintf(fileID,'elevation angle of target: %.2f \n',el_target(k));
                    fprintf(fileID,'reverberation parameter alpha: 0.50 \n');
                    fclose(fileID);


                    %add noise at different snrs
                    %calculate mbstoi, both full value and
                    %intermediate values
                    count_snr = count_snr + 1;
                end

                count = count + 1;
            end
        end
    end
        
    
end
        
        

