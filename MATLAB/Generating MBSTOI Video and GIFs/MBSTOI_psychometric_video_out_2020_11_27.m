%->either function with loop to generate audio data or read audio data

%have loop that does rolling frames of mbstoi and outputs a figure for each
%frame

%calculate rolling frames of MBSTOI
clear all;
close all;

addpath(genpath('MBSTOI'));
addpath(genpath('submodules'));
addpath(genpath('other_dependencies'));
addpath(genpath('RIR-Generator-master'));



%%%Goal of inputs is:

%Enter different speech samples (only one speaker will be chosen for any
%one test

%Enter different noise samples ordered (1,2,3,...,n)

%real or simulated impulse responses (1,0)
%If real impulse responses mode is chosen, a matrix of real impulse
%reponses need to be provided
%Enter room dimensions + (rt-60) If rt_60 = [], rt_60 will be chosen based on alpha = 0.5

%enter radii of src, this will also need to be seperated into radii for
%speaker and radii for noise

%Include Isotropic noise [1,0]

%Add plotting or no plotting

%Next step: Add this functionality and fix labelling of legend


%In the final scenario, not possible to use with graphics, the directional and
%isotropic noise should have their SNR controlled independently



[anechoic_speech,fs_s] = audioread('anechoic_speech.wav');
zero_append = zeros(56000-size(anechoic_speech,1),1);
anechoic_speech = cat(1,anechoic_speech,zero_append);
indiv_audio_length = size(anechoic_speech,1)/fs_s; %length of audio in seconds
noise_directory = 'D:\FYP\MATLAB\NOISEX-92';

%for noise_index = 1:2
audio_4_video = [];
video_frame_rate = 30;
audio_sample_rate = fs_s;

receiver_pos = [];
src_pos = [];
dimensions_x = [5,10,20,30,40,50];
dimensions_y = [5,10,20,30,40,50];
dimensions_z = [2,4,6,8,10,20];

video_file_name = 'video_test_room_dim.avi';
count_fig = 1;
total_frames = [];


for noise_index = 1:1
    noise_types = ["babble","factory1"];
    noise = load(strcat(noise_directory,'\',noise_types(1,noise_index))).(noise_types(1,noise_index));
    fs_n = 19980;
    noise = resample(noise,fs_s,fs_n);

    noise = noise(1:size(anechoic_speech,1),1);
    %if simulated
        for i = 1:1
        %for i = 1:size(dimensions_x,2)
            for j = 1:1
            %for j = 1:size(dimensions_y,2)
                for k = 1:1
                %for k = 1:size(dimensions_z,2)
                    dimensions = [dimensions_x(1,i),dimensions_y(1,j),dimensions_z(1,k)];

                    az_src = [0,90];
                    el_src = [0,0];
                    radii_src = [1.5,1.5];
                    rt60 = [];

                    [h_src, receiver_pos, src_pos] = gen_spatial_IR(az_src, el_src, radii_src, dimensions, rt60);

                    if size(h_src,1) < size(h_src,2)
                        h_src = permute(h_src,[1,3,2]);
                    end
                    speech_out_4ch = fftfilt(squeeze(h_src(1,:,:)),anechoic_speech);
                    noise_out_4ch = fftfilt(squeeze(h_src(2,:,:)),noise);

                    speech_out(:,1) = (speech_out_4ch(:,1) + speech_out_4ch(:,3))/2;
                    speech_out(:,2) = (speech_out_4ch(:,2) + speech_out_4ch(:,4))/2;

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
                            write_path = strcat('D:\FYP\MATLAB\MBSTOI\room_dim\','x_',int2str(i),'_y_',int2str(j),'_z_',int2str(k),'_snr_',int2str(snr_level),'.wav');
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


                        %enter number of noise sources
                        %number of noise sources is always number os sources -
                        %1
                        %Have that dictate the legend
                        %you will have names of noise types
                        %follow this page for info https://uk.mathworks.com/matlabcentral/answers/62393-adding-legend-in-a-plot-genereted-by-a-loop
                        R = size(receiver_pos,1);
                        S = size(src_pos,1);
                        h = subplot(1,2,2);
                        hold on;
                        for rr = 1:R
                            p_r(rr) = plot3(receiver_pos(rr,1),receiver_pos(rr,2),receiver_pos(rr,3),'xb','MarkerSize',12);
                        end
                        p_s(1) = plot3(src_pos(1,1),src_pos(1,2),src_pos(1,3),'or','MarkerSize',12);
                        for ss = 2:S
                            p_s(ss) = plot3(src_pos(ss,1),src_pos(ss,2),src_pos(ss,3),'dr','MarkerSize',12);
                        end
                        %plot3(sp_path(:,1),sp_path(:,2),sp_path(:,3),'r.');
                        legend([p_r(1),p_s({'receiver positions','speaker position','noise position'},'location','northwest');
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
                end
             end
        end
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

           
           
