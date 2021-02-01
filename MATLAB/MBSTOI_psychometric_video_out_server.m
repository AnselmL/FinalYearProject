%->either function with loop to generate audio data or read audio data

%have loop that does rolling frames of mbstoi and outputs a figure for each
%frame

%calculate rolling frames of MBSTOI
%clear all;
%close all;

addpath(genpath('MBSTOI'));
addpath(genpath('submodules'));
addpath(genpath('other_dependencies'));
addpath(genpath('RIR-Generator-master'));







[anechoic_speech,fs_s] = audioread('/home/al5517/Timit (subset) Speech dataset/extract/anechoic_speech.WAV');
%[anechoic_speech,fs_s] = audioread('/home/al5517/anechoic_speech.wav');
zero_append = zeros(56000-size(anechoic_speech,1),1);
anechoic_speech = cat(1,anechoic_speech,zero_append);
indiv_audio_length = size(anechoic_speech,1)/fs_s; %length of audio in seconds
noise_directory = '/home/al5517/NOISEX-92';

%for noise_index = 1:2
audio_4_video = [];
video_frame_rate = 30;
audio_sample_rate = fs_s;

receiver_pos = [];
src_pos = [];


video_file_name = '/home/al5517/output_videos/video_test_room_dim_full.avi';
count_fig = 1;
total_frames = [];

%%comments on data generation : training, testing set should include mix of
%anechoic and reverberant impulse responses

%%for both
%cell array of noise audio files    ...maybe even struct or something where
%they all have a name for labeling purposes
%cell array of target audio files   ...maybe even struct


%all I really need for this are the names of the timit speaker and the
%names of their audio files

%%%%%%%%%%%%%%Current idea%%%%%%%%%%%%%%%%%%%%%%%%%%

%have array for names of noise files to access and names of anechoic speech
%files to access

%This will require an organized directory

%have an array like 

%noise_name_array = ['NOISEX-92','factory']
%first column is noise directory, second column is file name in directory

%This will require a proper directory system which I would need to
%implement

%for anechoic speech
%speech_name_array = ['SA517','...']
%first column is name of TIMIT speaker, second column is name of voice line to use







%%for supplied impulse responses
%same concept as that of audio and noise arrays

%%for simulated
RIR_dir = 'E:\FYP\Room_IR_Database';
file_ext = '.wav';

az_target = [0];
el_target = [0];
az_noise = [90];
el_noise = [0];

%az_el_target = [0,0]; %this can also be matrix if multiple target azimuths and elevations are desired
%az_el_noise = [90,0]; %this can be matrix with column 1 for azimuth angle and column 2 for elevation angle
radii_target = [1,2,3];%row matrix of all desired target radii, specified in meters
radii_noise = [1,3,5]; %row matrix of all desired noise radii, specified in 

radii_total = unique(cat(2,radii_target,radii_noise));
az_total = unique(cat(2,az_target,az_noise));
el_total = unique(cat(2,el_target,el_noise));

dimensions = [5,5,4; 5,5,10;5,10,4; 10,10,4; 10,20,4; 20,20,4]; % room dimensions specified with [x,y,z] coordinates
rt60 = [];


%%boolean variables for functionality
graphing = 0; %specifies the inclusion of graph -> this is only for testing purposes and only works if simulated_IR is 1
simulated_IR = 1; %specifies whether or not simulated impulse responses should be generated (specifications for these are listed above)
isotropic_noise = 0; %specifies the inclusion of isotropic noise (only available for anechoic and simulated impulse response case)
no_IR = 0; %for case where no impulse response should be be used in audio generation

%%for next time -> modify part of the code from here downwards to fit with
%%the above specifications

%%goal for tonight: I want to have the data for anechoic speech with noise
%added 






for noise_index = 1:1
    noise_types = ["babble","factory1"];
    %noise = load(strcat(noise_directory,'/',noise_types(1,noise_index),'.mat')).(noise_types(1,noise_index));
    noise_str = load(strcat(noise_directory,'/',noise_types(1,noise_index),'.mat'));
    noise = noise_str.(noise_types(1,noise_index));
    fs_n = 19980;
    noise = resample(noise,fs_s,fs_n);

    noise = noise(1:size(anechoic_speech,1),1);
    %for i = 1:1
end
    
    
    
if simulated_IR
    for i = 1:size(dimensions,1)
        room_dim = dimensions(i,:);
        if graphing == 1
            [h_target, h_noise,receiver_pos,target_pos,noise_pos] = gen_spatial_IR_vectorised(az_target, el_target, az_noise, el_noise, radii_target, radii_noise, room_dim, rt60);
        elseif graphing == 0
            room_folder = strcat('x_',int2str(dimensions(i,1)),'_y_',int2str(dimensions(i,2)),'_z_',int2str(dimensions(i,3)));
            RIR_room_folder = strcat(RIR_dir,'\',room_folder);
            if ~exist(RIR_room_folder, 'dir')
                mkdir(RIR_room_folder)
            end
            for j = 1:size(radii_total,2)
                for k = 1:size(az_total,2)
                    for l = 1:size(el_total,2)
                        [h] = gen_spatial_IR(az_total(k), el_total(l), radii_total(j), room_dim, rt60);
                        if size(h,1) < size(h,2)
                            h = permute(h,[1,3,2]);
                        end
                        file_name = strcat('r_',int2str(radii_total(i)),'_az_',int2str(az_total(j)),'_el_',int2str(el_total(k)));
                        file_name_dir = strcat(RIR_room_folder,'\',file_name);
                        v_writewav(h,16000,file_name_dir,[],[],[],[]);
                    end
                end
            end
        end
    end
end

%things to remember, each target impulse response will be a new audio file
%all impulse responses provided will be used for each audio clip, always
%equal to the number of noise sources provided


for i = 1:size(dimensions,1)
    for j = 1:size(radii_target,2)
        for l = 1:size(az_target,2)
            for m = 1:size(el_target,2)
                for k = 1:size(radii_noise,2)
                    for n = 1:size(az_noise,2)
                        for o = 1:size(el_noise,2)
                            h_target_room_dir = strcat(RIR_dir,'\','x_',int2str(dimensions(i,1)),'_y_',int2str(dimensions(i,2)),'_z_',int2str(dimensions(i,3)));
                            h_target_wav_dir = strcat(h_target_room_dir,'\','r_',int2str(radii_target(j)),'_az_',int2str(az_target(l)),'_el_',int2str(el_target(m)),file_ext);
                            [h_target, h_fs] = audioread(h_target_wav_dir);

                            h_noise_room_dir = strcat(RIR_dir,'\','x_',int2str(dimensions(i,1)),'_y_',int2str(dimensions(i,2)),'_z_',int2str(dimensions(i,3)));
                            h_noise_wav_dir = strcat(h_noise_room_dir,'\','r_',int2str(radii_noise(j)),'_az_',int2str(az_noise(l)),'_el_',int2str(el_noise(m)),file_ext);
                            [h_noise, h_fs] = audioread(h_noise_wav_dir);
                            for p = 1:size(speech_sentences,2)
                                for q = 1:size(noise_types,2)

                                    
                                    
                                   
                                    
                                    
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

    speech_out_4ch = fftfilt(squeeze(h_src(1,:,:)),anechoic_speech);
    noise_out_4ch = fftfilt(squeeze(h_src(2,:,:)),noise);

    %generate stereo file for target
    speech_out(:,1) = (speech_out_4ch(:,1) + speech_out_4ch(:,3))/2; 
    speech_out(:,2) = (speech_out_4ch(:,2) + speech_out_4ch(:,4))/2;

    %generate stereo file for noise
    noise_out(:,1) = (noise_out_4ch(:,1) + noise_out_4ch(:,3))/2;
    noise_out(:,2) = (noise_out_4ch(:,2) + noise_out_4ch(:,4))/2;

    count_snr = 1;
    for snr_level = -30:3:30
        s_db=v_activlev(speech_out,16000,'d');  % speech level in dB
        mod_sound = 10^((-(s_db))/20)*speech_out;
        s_db=v_activlev(mod_sound,16000,'d');  % speech level in dB
        mod_sound = 10^((-(s_db))/20)*mod_sound;
        s_db = v_activlev(mod_sound,16000,'d');  % speech level in dB
        n_db = mean(pow2db(sum(noise_out.^2)/size(noise_out,1)));
        mixed_out = mod_sound + 10^(-(snr_level + n_db)/20)*noise_out;



        %elobes_writewav_wrapper(mixed_out,16000,write_path);

        if snr_level == 0
            write_path = strcat('/home/al5517/MBSTOI/room_dim/','x_',int2str(i),'_y_',int2str(j),'_z_',int2str(k),'_snr_',int2str(snr_level),'.wav');
            write_path = convertStringsToChars(write_path);
            v_writewav(mixed_out,16000,write_path,[],[],[],[]);
            mixed_out_write = mixed_out/max(max(abs(mixed_out)));
            audio_4_video = cat(1, audio_4_video, mixed_out_write);
        end


        xl = speech_out(:,1);
        xr = speech_out(:,2);
        %xl = anechoic_speech;
        %xr = anechoic_speech;
        yl = mixed_out(:,1);
        yr = mixed_out(:,2);

        mbstoi_mat(noise_index,i,j,k,count_snr) = mbstoi(xl,xr,yl,yr,fs_s);
        count_snr = count_snr + 1;
    end
for m = 1:(video_frame_rate*indiv_audio_length)
    f = figure('visible','off');
    set(gcf, 'WindowState', 'maximized');
    view_az = -38 + m*(1800/(video_frame_rate*indiv_audio_length));
    subplot(1,2,1);
    snrs = -30:3:30;
    hold on
        plot(snrs,squeeze(mbstoi_mat(1,i,j,k,:)));
        %plot(snrs,squeeze(mbstoi_circ(2,i,:)));
    hold off
    %title_string = sprintf('MBSTOI for static spatial sources with angle %3d degrees', angle);
    title_string = 'MBSTOI for different room dimensions';
    title(title_string);
    legend({'MBSTOI babble'},'location','northwest');
    %legend({'MBSTOI babble','MBSTOI factory'},'location', 'northwest');
    ylim([0 1]);
    xlabel('SNR [dB]');
    ylabel('SIP score');


    %frame = getframe(h); 

    %im = frame2im(frame);


    R = size(receiver_pos,1);
    S = size(src_pos,1);
    h = subplot(1,2,2);
    hold on;
    for rr = 1:R
        plot3(receiver_pos(rr,1),receiver_pos(rr,2),receiver_pos(rr,3),'xb','MarkerSize',12);
    end
    plot3(src_pos(1,1),src_pos(1,2),src_pos(1,3),'or','MarkerSize',12)
    for ss = 2:S
        plot3(src_pos(ss,1),src_pos(ss,2),src_pos(ss,3),'dr','MarkerSize',12)
    end
    %plot3(sp_path(:,1),sp_path(:,2),sp_path(:,3),'r.');
    legend({'receiver positions','speaker position','noise position'},'location','northwest');
    axis([0 dimensions_x(1,i) 0 dimensions_y(1,j) 0 dimensions_z(1,k)]);
    pbaspect([1,1,1]);
    h.View = [view_az,20];
    grid on;
    box on;
    axis square;
    hold off;

    %get(h)


    %g = figure(count_fig);
    count_fig = count_fig + 1;
    frame = getframe(f);
    im = frame2im(frame); 
    %frame_copies = repmat(im,[1 1 1 indiv_audio_length*video_frame_rate]);
    total_frames = cat(4,total_frames,im);

    %close all;
end
                
%{
for i = 1:size(mbstoi_mat,2)
    for j = 1:size(mbstoi_mat,3)
        for k = 1:size(mbstoi_mat,4)
            for m = 1:(video_frame_rate*indiv_audio_length)
                figure(count_fig);
                set(gcf, 'WindowState', 'maximized');
                view_az = -38 + m*(360/video_frame_rate*indiv_audio_length);
                subplot(1,2,1);
                snrs = -30:3:30;
                hold on
                    plot(snrs,squeeze(mbstoi_mat(1,i,j,k,:)));
                    %plot(snrs,squeeze(mbstoi_circ(2,i,:)));
                hold off
                %title_string = sprintf('MBSTOI for static spatial sources with angle %3d degrees', angle);
                title_string = 'MBSTOI for different room dimensions';
                title(title_string);
                legend({'MBSTOI babble'},'location','northwest');
                %legend({'MBSTOI babble','MBSTOI factory'},'location', 'northwest');
                ylim([0 1]);
                xlabel('SNR [dB]');
                ylabel('SIP score');


                %frame = getframe(h); 

                %im = frame2im(frame);


                R = size(receiver_pos,1);
                S = size(src_pos,1);
                h = subplot(1,2,2);
                hold on;
                for rr = 1:R
                    plot3(receiver_pos(rr,1),receiver_pos(rr,2),receiver_pos(rr,3),'x','MarkerSize',12);
                end
                plot3(src_pos(1,1),src_pos(1,2),src_pos(1,3),'or','MarkerSize',12)
                for ss = 2:S
                    plot3(src_pos(ss,1),src_pos(ss,2),src_pos(ss,3),'dr','MarkerSize',12)
                end
                %plot3(sp_path(:,1),sp_path(:,2),sp_path(:,3),'r.');
                legend({'receiver positions','speaker position','noise position'},'location','northwest');
                axis([0 dimensions_x(1,i) 0 dimensions_y(1,j) 0 dimensions_z(1,k)]);
                pbaspect([1,1,1]);
                h.View = [view_az,20];
                grid on;
                box on;
                axis square;
                hold off;

                %get(h)


                g = figure(count_fig);
                count_fig = count_fig + 1;
                frame = getframe(g);
                im = frame2im(frame); 
                %frame_copies = repmat(im,[1 1 1 indiv_audio_length*video_frame_rate]);
                total_frames = cat(4,total_frames,im);
                

            end
        end
    end
end
%}

if ~eq(size(audio_4_video,1)/(size(total_frames,4)/video_frame_rate),audio_sample_rate)
    error('length of audio file does not match length of video with current sample rate');
end

videoFWriter = vision.VideoFileWriter(video_file_name,...
                                      'AudioInputPort', true, ...
                                      'FrameRate',  video_frame_rate);
videoFWriter.VideoCompressor = 'MJPEG Compressor';
videoFWriter.AudioCompressor = 'None (uncompressed)';

audio_4_video = resample(audio_4_video,video_frame_rate*10000,audio_sample_rate);
audio_sample_rate = video_frame_rate*10000;
for i = 1:size(total_frames,4)
   fprintf('Frame: %d/%d\n', i, size(total_frames,4));
   index_start = round((i-1)*(1/video_frame_rate)*audio_sample_rate + 1);
   index_end = round(i*(1/video_frame_rate)*audio_sample_rate);
   step(videoFWriter, squeeze(total_frames(:,:,:,i)), audio_4_video(index_start: index_end,:)); 
end
release(videoFWriter);










































