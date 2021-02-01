%->either function with loop to generate audio data or read audio data

%have loop that does rolling frames of mbstoi and outputs a figure for each
%frame

%calculate rolling frames of MBSTOI
%snrs = -30:3:30;
clear all;
close all;

addpath(genpath('MBSTOI'));
addpath(genpath('submodules'));
addpath(genpath('other_dependencies'));

%dbstoi_mat = zeros(188,17);
%mbstoi_mat = zeros(188,17);

%nb_sources = ;
%src_pos = 




[clean_file,fs_clean] = audioread("clean_courtyard.wav");
if size(clean_file,1) < size(clean_file,2)
    clean_file = clean_file.';
end
noisy_file_directory = "E:\IP\IN_PREPROCESSING\NewDataset\model_sep\cafeteria";
nb_subdirec = 17;

for j=nb_subdirec:-1:1
    wav_directory = strcat(noisy_file_directory,'\',int2str(j),'\','mixed.wav');
    [noisy_file,fs_noisy] = audioread(wav_directory);
    if fs_noisy ~= fs_clean
        error('sampling frequencies do not match');
    end
    if size(noisy_file,1) < size(noisy_file,2)
        noisy_file = noisy_file.';
    end
    
   if size(noisy_file,2) > 2 && size(noisy_file,2) == 4
       yl = (noisy_file(:,1) + noisy_file(:,3))/2;
       yr = (noisy_file(:,2) + noisy_file(:,4))/2;
   elseif size(noisy_file,2) == 2
       yl = noisy_file(:,1);
       yr = noisy_file(:,2);
   else 
       error("noisy file is neither made up of 2 nor 4-channels");
   end
   if size(clean_file,2) > 2 && size(clean_file,2) == 4
       xl = (clean_file(:,1) + clean_file(:,3))/2;
       xr = (clean_file(:,2) + clean_file(:,4))/2;
   elseif size(clean_file,2) == 2
       xl = clean_file(:,1);
       xr = clean_file(:,2);
   elseif size(clean_file,2) == 1
       xl = clean_file(:);
       xr = clean_file(:);
   else 
       error("clean file is neither made up of 1, nor 2, nor 4-channels");
   end
    i_prev = 1;
    step = 0.525*fs_noisy;
    count = 1;
    for i = 1:0.1*fs_noisy:300800 %(size(noisy_file,1)-0.4*fs_noisy)
       % Correlated speech
       
       index_start = i;
       index_end = i + step;
       %x_l = cat(1,xl(index_start:index_end),zeros(10035 - (index_end-index_start),1));
       %x_r = cat(1,xr(index_start:index_end),zeros(10035 - (index_end-index_start),1));
       %y_l = cat(1,yl(index_start:index_end),zeros(10035 - (index_end-index_start),1));
       %y_r = cat(1,yr(index_start:index_end),zeros(10035 - (index_end-index_start),1));
       dbstoi_mat(count,j) = dbstoi(xl(index_start:index_end,1),xr(index_start:index_end,1),yl(index_start:index_end,1),yr(index_start:index_end,1),fs_noisy);
       mbstoi_mat(count,j) = mbstoi(xl(index_start:index_end,1),xr(index_start:index_end,1),yl(index_start:index_end,1),yr(index_start:index_end,1),fs_noisy);
       %dbstoi_mat(count,j) = dbstoi(x_l,x_r,y_l,y_r,fs_noisy);
       %mbstoi_mat(count,j) = mbstoi(x_l,x_r,y_l,y_r,fs_noisy);
       count = count + 1;
    end
end

%{
[anechoic_speech,fs_s] = audioread('anechoic_speech.wav');
zero_append = zeros(56000-size(anechoic_speech,1),1);
anechoic_speech = cat(1,anechoic_speech,zero_append);
indiv_audio_length = size(anechoic_speech,1)/fs_s; %length of audio in seconds
noise_directory = 'D:\FYP\MATLAB\NOISEX-92';
%{
for noise_index = 1:4
    %noise_directory = 'D:\FYP\MATLAB\NOISEX-92';
    noise_types = ["babble","factory1","machinegun", "white"];

    noise = load(strcat(noise_directory,'\',noise_types(1,noise_index))).(noise_types(1,noise_index));
    fs_n = 19980;
    noise = resample(noise,fs_s,fs_n);

    noise = noise(1:size(anechoic_speech,1),1);

    %noise_index = 1 -> babble
    %noise_index = 2 -> factory1
    %babble, factory1, machinegun, white
   for impulse_response_index = 1:2

       if impulse_response_index == 1
           angle = -90; %degrees
           distance = 80; %cm
           noise_out = gen_1speaker_BTE(noise,angle,distance);
           speech_out = gen_1speaker_BTE(anechoic_speech,0,300);
       elseif impulse_response_index == 2
           hrir_directory = 'D:\IP\Kayser2009\HRIR_database_wav\hrir\cafeteria\cafeteria';
           noise_pos = '_1_C.wav';
           speaker_pos = '_1_A.wav';
           [noise_hrir, fs_hrir] = audioread(strcat(hrir_directory,noise_pos));
           [speaker_hrir, fs_hrir] = audioread(strcat(hrir_directory,speaker_pos));
           noise_hrir = noise_hrir(:,[3,4,7,8]);
           speaker_hrir = speaker_hrir(:,[3,4,7,8]);
           
           noise_out_4ch = fftfilt(noise_hrir,noise);
           speech_out_4ch = fftfilt(speaker_hrir,anechoic_speech);
           
           noise_out_4ch = noise_out_4ch/max(max(abs(noise_out_4ch)));
           speech_out_4ch = speech_out_4ch/max(max(abs(speech_out_4ch)));
           
           noise_out(:,1) = (noise_out_4ch(:,1) + noise_out_4ch(:,3))/2;
           noise_out(:,2) = (noise_out_4ch(:,2) + noise_out_4ch(:,4))/2;
           speech_out(:,1) = (speech_out_4ch(:,1) + speech_out_4ch(:,3))/2;
           speech_out(:,2) = (speech_out_4ch(:,2) + speech_out_4ch(:,4))/2;
       end
       count = 1;
       for snr_level = -30:3:30

        s_db=v_activlev(speech_out,16000,'d');  % speech level in dB
        mod_sound = 10^((-(s_db))/20)*speech_out;
        s_db=v_activlev(mod_sound,16000,'d');  % speech level in dB
        mod_sound = 10^((-(s_db))/20)*mod_sound;
        s_db = v_activlev(mod_sound,16000,'d');  % speech level in dB
        n_db = mean(pow2db(sum(noise_out.^2)/size(noise_out,1)));
        mixed_out = mod_sound + 10^(-(snr_level + n_db)/20)*noise_out;
        
        write_path = strcat('D:\FYP\MATLAB\MBSTOI\static_audio\',noise_types(1,noise_index),'\',int2str(impulse_response_index),'\',int2str(snr_level),'.wav');
        %write_path_2 = 'D:\FYP\MATLAB\MBSTOI\static_audio\babble\1\-30.wav';
        write_path = convertStringsToChars(write_path);
        %disp(write_path_1);
        %disp(write_path_2);
        %elobes_writewav_wrapper(mixed_out,16000,write_path);
        v_writewav(mixed_out,16000,write_path,[],[],[],[]);
        

        xl = speech_out(:,1);
        xr = speech_out(:,2);
        yl = mixed_out(:,1);
        yr = mixed_out(:,2);
        
        mbstoi_static(noise_index,impulse_response_index,count) = mbstoi(xl,xr,yl,yr,fs_s);
        count = count + 1;
       end
   end
end
%}
%{
figure(1);
snrs = -30:3:30;
hold on
    plot(snrs,squeeze(mbstoi_static(1,1,:)));
    plot(snrs,squeeze(mbstoi_static(2,1,:)));
    plot(snrs,squeeze(mbstoi_static(3,1,:)));
    plot(snrs,squeeze(mbstoi_static(4,1,:)));
hold off
title_string = strcat('MBSTOI for static spatial sources with angle -90 degrees (anechoic)');
title(title_string);
legend({'MBSTOI babble','MBSTOI factory', 'MBSTOI machinegun', 'MBSTOI white'},'location', 'northwest');
xlabel('SNR [dB]');
ylabel('SIP score')

figure(2);
snrs = -30:3:30;
hold on
    plot(snrs,squeeze(mbstoi_static(1,1,:)));
    plot(snrs,squeeze(mbstoi_static(1,2,:)));
    plot(snrs,squeeze(mbstoi_static(2,2,:)));
    plot(snrs,squeeze(mbstoi_static(3,2,:)));
    plot(snrs,squeeze(mbstoi_static(4,2,:)));
hold off
ylim([0 1]);
title_string = strcat('MBSTOI for static spatial sources with angle -90 degrees (reverberant)');
title(title_string);
legend({'MBSTOI babble (anechoic)','MBSTOI babble','MBSTOI factory', 'MBSTOI machinegun', 'MBSTOI white'},'location', 'northwest');
xlabel('SNR [dB]');
ylabel('SIP score')


%print(figure(1),'MBSTOI_static_anechoic','-depsc');
print(figure(1),'MBSTOI_static_anechoic','-dpng');
%print(figure(2),'MBSTOI_static_reverberant','-depsc');
print(figure(2),'MBSTOI_static_reverberant','-dpng');

%}

%for noise_index = 1:2
audio_4_video = [];
video_frame_rate = 10;
audio_sample_rate = fs_s;
for noise_index = 1:1
    noise_types = ["babble","factory1"];
    noise = load(strcat(noise_directory,'\',noise_types(1,noise_index))).(noise_types(1,noise_index));
    fs_n = 19980;
    noise = resample(noise,fs_s,fs_n);

    noise = noise(1:size(anechoic_speech,1),1);
    i = 1;
    for angle = -175:5:180
           angle_s = 0; %degrees
           distance_n = 80; %cm
           distance_s = 300;
           noise_out = gen_1speaker_BTE(noise,angle,distance_n);
           speech_out = gen_1speaker_BTE(anechoic_speech,angle_s,distance_s);
           count = 1;
           for snr_level = -30:3:30

            s_db=v_activlev(speech_out,16000,'d');  % speech level in dB
            mod_sound = 10^((-(s_db))/20)*speech_out;
            s_db=v_activlev(mod_sound,16000,'d');  % speech level in dB
            mod_sound = 10^((-(s_db))/20)*mod_sound;
            s_db = v_activlev(mod_sound,16000,'d');  % speech level in dB
            n_db = mean(pow2db(sum(noise_out.^2)/size(noise_out,1)));
            mixed_out = mod_sound + 10^(-(snr_level + n_db)/20)*noise_out;
            
            write_path = strcat('D:\FYP\MATLAB\MBSTOI\circ_audio\',noise_types(1,noise_index),'\',int2str(angle),'\',int2str(snr_level),'.wav');
            write_path = convertStringsToChars(write_path);
            %elobes_writewav_wrapper(mixed_out,16000,write_path);
            
            if snr_level == 0
                audio_4_video = cat(1, audio_4_video, mixed_out/max(max(abs(audio_4_video))));
            end
            v_writewav(mixed_out,16000,write_path,[],[],[],[]);

            xl = speech_out(:,1);
            xr = speech_out(:,2);
            yl = mixed_out(:,1);
            yr = mixed_out(:,2);

            mbstoi_circ(noise_index,i,count) = mbstoi(xl,xr,yl,yr,fs_s);
            count = count + 1;
           end
           i = i + 1;
    end
end
%}
%filename = 'MBSTOI_static_changing_angle.gif';
video_file_name = 'video_test.avi';
%{
total_frames = [];
for i = 1:size(mbstoi_circ,2)
    angle = -175 + (i-1)*5;
    h = figure(i);
    snrs = -30:3:30;
    hold on
        plot(snrs,squeeze(mbstoi_circ(1,i,:)));
        %plot(snrs,squeeze(mbstoi_circ(2,i,:)));
    hold off
    title_string = sprintf('MBSTOI for static spatial sources with angle %3d degrees', angle);
    title(title_string);
    legend({'MBSTOI babble'},'location','northwest');
    %legend({'MBSTOI babble','MBSTOI factory'},'location', 'northwest');
    ylim([0 1]);
    xlabel('SNR [dB]');
    ylabel('SIP score')
    
    
    frame = getframe(h);
    im = frame2im(frame); 
    frame_copies = repmat(im,[1 1 1 indiv_audio_length*video_frame_rate]);
    total_frames = cat(4,total_frames,frame_copies);

    %{
    [imind,cm] = rgb2ind(im,256);
    % Write to the GIF File 
    if i == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
    else 
        imwrite(imind,cm,filename,'gif','DelayTime',0.25,'WriteMode','append'); 
    end 
    %}
end


if ~eq(size(audio_4_video,1)/(size(total_frames,4)/video_frame_rate),audio_sample_rate)
    error('length of audio file does not match length of video with current sample rate');
end

%}
%{
videoFWriter = vision.VideoFileWriter(video_file_name,...
                                      'AudioInputPort', true, ...
                                      'FrameRate',  video_frame_rate);
videoFWriter.VideoCompressor = 'MJPEG Compressor';
videoFWriter.AudioCompressor = 'None (uncompressed)';
for i = 1:size(total_frames,4)
   fprintf('Frame: %d/%d\n', i, size(total_frames,4));
   index_start = round((i-1)*(1/video_frame_rate)*audio_sample_rate + 1);
   index_end = round(i*(1/video_frame_rate)*audio_sample_rate);
   step(videoFWriter, squeeze(total_frames(:,:,:,i)), audio_4_video(index_start: index_end,:)); 
end
release(videoFWriter);
%}
           
           
    

       %impulse_response_index (IRI = 1 -> anechoic, IRI = 2 ->
       %cafeteria)
       %(2 sets: anechoic, cafeteria
           
     
%other loop for 2 sets of noises in different positions using anechoic data

%babble and factory



%graphing
%{
filename = 'MBSTOI_DBSTOI_cafeteria.gif';
for i = 1:size(dbstoi_mat,1)
    h = figure(i);
    snrs = -30:3:30;
    hold on
        plot(snrs,dbstoi_mat(i,:));
        plot(snrs,mbstoi_mat(i,:));
    hold off
    title('DBSTOI/MBSTOI');
    legend({'DBSTOI','MBSTOI'},'location', 'northwest');
    xlabel('SNR [dB]');
    ylabel('SIP score')
    
  frame = getframe(h); 
  im = frame2im(frame); 
  [imind,cm] = rgb2ind(im,256); 
  % Write to the GIF File 
  if i == 1 
      imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
  else 
      imwrite(imind,cm,filename,'gif','DelayTime',0.5,'WriteMode','append'); 
  end 
end

%}



%These figures should be subfigures where one side is the psychometric
%curve and the other side is the audio diagram of what's happening (taken
%from format of speaker visualization from placement. Have one extra row
%have a moving red rectangle based on where in the audio clip we currently
%are

%convert figures to gif
