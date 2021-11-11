%script written by A. Lohmann as part the MEng Final Year ProjectHRIR_base

%script to generate training data files with BRIR simulation (this is what
%was used for the final project)

%All of the functions were unrolled such as to improve speed (generating
%2-3 TIMIT folds already takes 24-48 hours as is).

%the naming of variables is not the best, conventions may not be followed.
%Hopefully it's still readable.
clear all;

%adding paths for files, you will likely need to change the directories
addpath(genpath('E:\FYP\MATLAB\MBSTOI'));
addpath(genpath('E:\FYP\MATLAB\submodules'));
addpath(genpath('E:\FYP\MATLAB\other_dependencies'));
addpath(genpath('E:\FYP\MATLAB\RIR-Generator-master'));
addpath(genpath('E:\FYP\MATLAB\ANF-Generator-master'));


%the base directory
base_dir = 'E:\FYP\MATLAB';

%the directory where HRTF/HRIR files are stored (I have tried replacing the
%majority of place HRTF was used for HRIR as that the correct version. In
%certain places I have left hrtf because that is what the files will be
%called. If there is any bug in the code, the naming here is likely the
%place to look.)
HRIR_base_dir = 'E:\FYP\MATLAB\HRTF';

%the directory in which the training audio files will be saved
out_base_dir = 'E:\FYP\feature_preprocessing_input\directional_2021_11_11\TRAIN';


%creating cell array containing TIMIT file wav directories
TIMIT_wav_dirs = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_train;

%creating cell array containing noise file wave directories
noise_wav_dirs = load('TUT_wav_dirs').TUT_wav_dirs;


%different microphone positions found in all HRIRs used in the training set
mic_pos_SCUT = [-0.0900,0,0;
                 0.0900,0,0];
             
mic_pos_BKwHA_bte_array = [-0.0800,-0.0136,  0.0374;
            0.0800,-0.0136,0.0374;
            -0.0800,-0.0285,0.0327;
            0.0800,-0.0285,0.0327];

mic_pos_BKwHA_bte = [(mic_pos_BKwHA_bte_array(1,:) + mic_pos_BKwHA_bte_array(3,:))/2;
                        (mic_pos_BKwHA_bte_array(2,:) + mic_pos_BKwHA_bte_array(4,:))/2];

mic_pos_ARI_bte = [-0.0900,0,0;
                    0.0900,0,0];

mic_pos_FF_KOELN = [-0.0875,0,0;
                    0.0875,0,0];

%the 'pools' for all the sampled parameters (see FYP report). This allows
%pseudorandom data generation in order to create an asymmetric dataset
TIMIT_wav_pool = [];
noise_index_pool = [];
SNR_diff_index_pool = [];
SNR_direc_index_pool = [];
az_index_pool = [];
el_index_pool = [];
alpha_add_index_pool = [];
alpha_index_pool = [];
x_room_dim_index_pool = [];
y_room_dim_index_pool = [];
z_room_dim_index_pool = [];
radius_ratio_index_pool = [];
HRIR_type_pool = [];
direc_pool = [];
direc_type_pool = [];

%how many TIMIT folds you want (each fold is one run-through the entire
%TIMIT set)
nb_folds = 3;

%the choice of room dimensions for your RIR simulations
dimension_x = 3:10;
dimension_y = 3:15;
dimension_z = 3:7;

%a set of common room materials and their absorption coefficients
%NOTE: the RIR generator used in this project is a modified version of the
%Habets RIR generator you can find on his website
alpha_plasterboard_ceilling = 0.0983;
alpha_wood_ceilling = 0.147;

alpha_plaster = 0.0283;
alpha_plywood = 0.2;

alpha_carpet = 0.182;
alpha_wood_f = 0.05;

%arrays specifiying the what the full 'pools' are like
alpha_add = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7];

HRIR_type = 1:4;

alpha_array = [1, 1, 1, 1, 1, 1;
               1, 1, 1, 1, 1, 1;
               1, 1, 1, 1, 1, 1;
         alpha_plaster, alpha_plaster, alpha_plaster, alpha_plaster,alpha_carpet, alpha_plasterboard_ceilling;
         alpha_plaster, alpha_plaster, alpha_plaster, alpha_plaster,alpha_carpet, alpha_wood_ceilling;
         alpha_plaster, alpha_plaster, alpha_plaster, alpha_plaster,alpha_wood_f, alpha_plasterboard_ceilling;
         alpha_plaster, alpha_plaster, alpha_plaster, alpha_plaster,alpha_wood_f, alpha_wood_ceilling;
         alpha_plywood, alpha_plywood, alpha_plywood, alpha_plywood,alpha_carpet, alpha_plasterboard_ceilling;
         alpha_plywood, alpha_plywood, alpha_plywood, alpha_plywood,alpha_carpet, alpha_wood_ceilling;
         alpha_plywood, alpha_plywood, alpha_plywood, alpha_plywood,alpha_wood_f, alpha_plasterboard_ceilling;
         alpha_plywood, alpha_plywood, alpha_plywood, alpha_plywood,alpha_wood_f, alpha_wood_ceilling];
     
rt60 = [];
az_target = 0:5:355;
el_target = [-20,-15,-10,-5,0,5,10,15,20];
radii_target_ratio = 0.25:0.05:0.75;
SNR_diff = -9:3:30;
SNR_direc = -6:3:15;
direc_type = [0,1]; %0 -> noise source, 1 -> speech
direc_on = [0,1,2]; %0 -> no directional interfer, 1 -> 1 directional interfer, 2 -> 2 directional interferers

%additional variables for pseudorandom index allocation that need to be in a
%different format
radius_target_direc = zeros(1,2);
h_omni_direc = cell(1,2);
direc_type_index = cell(1,2);
SNR_direc_index = cell(1,2);
HRIR_dir_direc = cell(1,2);
HRIR_out_dir_direc = cell(1,2);
HRIR_direc = cell(1,2);
BRIR_direc = cell(1,2);
noise_direc_dir = cell(1,2);
TIMIT_direc_dir = cell(1,2);

%what file we are currently on
file_counter = 1;
%for each fold
for fold = 1:nb_folds
    %at each fold we will have run out of TIMIT wavs, so we need to refill
    %the pool
    if isempty(TIMIT_wav_pool)
        TIMIT_wav_pool = 1:size(TIMIT_wav_dirs,1);
    end
    %while the pool still has some parameters inside it
    while(~isempty(TIMIT_wav_pool))
        %fill up the pools if they are empty
        if isempty(noise_index_pool)
            noise_index_pool = 1:size(noise_wav_dirs,1);
        end
        if isempty(SNR_diff_index_pool)    
            SNR_diff_index_pool = 1:length(SNR_diff);
        end
        if isempty(SNR_direc_index_pool)    
            SNR_direc_index_pool = 1:length(SNR_direc);
        end
        if isempty(az_index_pool)
            az_index_pool = 1:length(az_target);
        end
        if isempty(el_index_pool)
            el_index_pool = 1:length(el_target);
        end
        if isempty(alpha_add_index_pool)
            alpha_add_index_pool = 1:length(alpha_add);
        end
        if isempty(alpha_index_pool)
            alpha_index_pool = 1:size(alpha_array,1);
        end
        if isempty(x_room_dim_index_pool)
            x_room_dim_index_pool = 1:length(dimension_x);
        end
        if isempty(y_room_dim_index_pool)
            y_room_dim_index_pool = 1:length(dimension_y);
        end
        if isempty(z_room_dim_index_pool)
            z_room_dim_index_pool = 1:length(dimension_z);
        end
        if isempty(radius_ratio_index_pool)
            radius_ratio_index_pool = 1:length(radii_target_ratio);
        end
        if isempty(direc_pool)
            direc_pool = 1:3;
        end
        if isempty(direc_type_pool)
            direc_type_pool = 1:2;
        end
        if isempty(HRIR_type_pool)
            HRIR_type_pool = 1:length(HRIR_type);
        end

        %picking pseudorandom index
        TIMIT_wav_index = TIMIT_wav_pool(randperm(size(TIMIT_wav_pool,2),1));
        %removing index from pool
        TIMIT_wav_pool=TIMIT_wav_pool(TIMIT_wav_pool~=TIMIT_wav_index);
        %allocating specific directory for anechoic speech file
        anechoic_speech_dir = TIMIT_wav_dirs{TIMIT_wav_index,1};
        %reading audio file
        [anechoic_speech, fs_speech] = audioread(anechoic_speech_dir);

        %picking all of the pseudorandom indexes required for RIR
        %simulation and removing them from the pools
        alpha_index = alpha_index_pool(randperm(size(alpha_index_pool,2),1));
        alpha_index_pool=alpha_index_pool(alpha_index_pool~=alpha_index);

        alpha_add_index = alpha_add_index_pool(randperm(size(alpha_add_index_pool,2),1));
        alpha_add_index_pool=alpha_add_index_pool(alpha_add_index_pool~=alpha_add_index);

        x_room_dim_index = x_room_dim_index_pool(randperm(size(x_room_dim_index_pool,2),1));
        x_room_dim_index_pool=x_room_dim_index_pool(x_room_dim_index_pool~=x_room_dim_index);

        y_room_dim_index = y_room_dim_index_pool(randperm(size(y_room_dim_index_pool,2),1));
        y_room_dim_index_pool=y_room_dim_index_pool(y_room_dim_index_pool~=y_room_dim_index);

        z_room_dim_index = z_room_dim_index_pool(randperm(size(z_room_dim_index_pool,2),1));
        z_room_dim_index_pool=z_room_dim_index_pool(z_room_dim_index_pool~=z_room_dim_index);

        room_dimension = [dimension_x(x_room_dim_index),dimension_y(y_room_dim_index),dimension_z(z_room_dim_index)];

        alpha_vals = alpha_array(alpha_index,:) + alpha_add(alpha_add_index);
        %some of the alpha values were set to 1 to force some training
        %samples to solely utilize the HRIR, in this case no RIR is
        %simulated
        if ~any(alpha_vals >= 1)
            
            %picking and removing values from pools
            az_index = az_index_pool(randperm(size(az_index_pool,2),1));
            az_index_pool=az_index_pool(az_index_pool~=az_index);

            el_index = el_index_pool(randperm(size(el_index_pool,2),1));
            el_index_pool=el_index_pool(el_index_pool~=el_index);

            radius_ratio_index = radius_ratio_index_pool(randperm(size(radius_ratio_index_pool,2),1));
            radius_ratio_index_pool=radius_ratio_index_pool(radius_ratio_index_pool~=radius_ratio_index);
            
            %correcting azimuth angle to be in the proper format for the
            %RIR generator
            az_target_rir_gen = az_target(az_index) + 90;
            if az_target_rir_gen > 180
                az_target_rir_gen = az_target_rir_gen - 360;
            end
            %generating simulated RIR
            [h_omni, radius_target,max_radius_target] = gen_spatial_IR_2021_05_27(az_target_rir_gen, el_target(1,el_index), radii_target_ratio(radius_ratio_index), room_dimension, alpha_vals);
            
            %selecting HRIR that is the best match
            if radius_target < 0.45
                HRIR_subdir = 'SCUT';
                radius_HRIR = 0.4;
            elseif radius_target >= 0.45 && radius_target < 0.55
                HRIR_subdir = 'SCUT';
                radius_HRIR = 0.5;
            elseif radius_target >= 0.55 && radius_target < 0.65
                HRIR_subdir = 'SCUT';
                radius_HRIR = 0.6;
            elseif radius_target >= 0.65 && radius_target < 0.75
                HRIR_subdir = 'SCUT';
                radius_HRIR = 0.7;
            elseif radius_target >= 0.75 && radius_target < 0.85
                HRIR_subdir = 'SCUT';
                radius_HRIR = 0.8;
            elseif radius_target >= 0.85 && radius_target < 0.95
                HRIR_subdir = 'SCUT';
                radius_HRIR = 0.9;
            elseif radius_target >= 0.95 && radius_target < 1.1
                HRIR_subdir = 'SCUT';
                radius_HRIR = 1;
            elseif radius_target >= 1.1 && radius_target < 1.45
                HRIR_subdir = 'ARI_bte';
                radius_HRIR = 1.2;
            elseif radius_target >= 1.45 && radius_target < 2.7
                HRIR_subdir = 'MMHR-HRTF\BKwHA';
                radius_HRIR = 1.7;
            elseif radius_target >= 2.7
                HRIR_subdir = 'FF_KOELN';
                radius_HRIR = 3.25;
            else
                error('invalid radius');
            end
            %SCUT database is limited in its available elevation angles, so
            %this little bit of code allocates the closest elevation angle
            switch HRIR_subdir
                case 'SCUT'
                if el_target(el_index) ~= -15 || el_target(el_index) ~= 0 || el_target(el_index) ~= 15
                    dist(1) = abs(el_target(el_index) + 15);
                    dist(2) = abs(el_target(el_index));
                    dist(3) = abs(el_target(el_index) - 15);
                    [~,min_index] = min(dist);
                    if min_index == 1
                        HRIR_el = -15;
                    elseif min_index == 2
                        HRIR_el = 0;
                    elseif min_index == 3
                        HRIR_el = 15;
                    else
                        error('wrong min index');
                    end
                end
                otherwise
                    HRIR_el = el_target(el_index);
            end
                   
            %building the appropriate HRIR directory based on the chosen parameters        
            HRIR_dir = strcat('az_',num2str(az_target(az_index)),'_el_',num2str(HRIR_el),'_radius_',num2str(radius_HRIR));
            
            HRIR_out_dir = strcat(HRIR_base_dir,'\',HRIR_subdir,'\',HRIR_dir,'.wav');

            %reading HRIR 
            [HRIR,fs_HRIR] = audioread(HRIR_out_dir);
            %making sure HRIR is in correct format
            if size(HRIR,1) < size(HRIR,2)
                HRIR = HRIR.';
            end
            %resampling (not always necessary)
            HRIR = resample(HRIR,fs_speech,fs_HRIR);
            %generating BRIR
            BRIR = fftfilt(HRIR,h_omni);

        else
            %picking and removing HRIR parameter from pool
            HRIR_index = HRIR_type_pool(randperm(size(HRIR_type_pool,2),1));
            HRIR_type_pool=HRIR_type_pool(HRIR_type_pool~=HRIR_index);
            
            %choosing HRIR subdir based on parameter
            if HRIR_index == 1
                HRIR_subdir = 'SCUT';
            elseif HRIR_index == 2
                HRIR_subdir = 'ARI_bte';
            elseif HRIR_index == 3
                HRIR_subdir = 'MMHR-HRIR\BKwHA';
            elseif HRIR_index == 4
                HRIR_subdir = 'FF_KOELN';
            end
            %here the HRIR is simply selected randomly (not pseudorandom)
            HRIR_subdir_dir = strcat(HRIR_base_dir,'\',HRIR_subdir);
            file_type = '.wav';
            wav_files = dir(strcat(HRIR_subdir_dir,'\','*',file_type));
            wav_files = {wav_files.name};
            nb_files = length(wav_files);
            
            rand_file_ind = randi(nb_files);

            HRIR_out_dir = strcat(HRIR_subdir_dir,'\',wav_files{rand_file_ind});
            [HRIR,fs_HRIR] = audioread(HRIR_out_dir);
            if size(HRIR,1) < size(HRIR,2)
                HRIR = HRIR.';
            end
            HRIR = resample(HRIR,fs_speech,fs_HRIR);

            BRIR = HRIR;
        end
        %generating binaural speech
        speech_binaural = fftfilt(BRIR, anechoic_speech);
        
        %picking parameters for noise selection
        noise_index = noise_index_pool(randperm(size(noise_index_pool,2),1));
        noise_index_pool = noise_index_pool(noise_index_pool~=noise_index);
        
        noise_dir = noise_wav_dirs{noise_index};
        %reading noise file
        [noise_wav,fs_noise] = audioread(noise_dir);
        %resampling
        if fs_noise ~= fs_speech
            noise_wav = resample(noise_wav,fs_speech,fs_noise);
            fs_noise = fs_speech;
        end
        %creating empty shell for mono noise (noise needs to be mono before
        %using diffuse noise generator
        noise_mono = zeros(size(noise_wav,1),1);
        %converting noise to mono
        for chan = 1:size(noise_wav,2)
            noise_mono = noise_mono + noise_wav(:,chan);
        end
        %correcting size to make sure the file is big enough given the
        %length of the anechoic file (requirement of the diffuse noise
        %generator script)
        if 2*size(anechoic_speech,1) > length(noise_mono)
            noise_mono_temp = [noise_mono;noise_mono];
            noise_mono = noise_mono_temp;
        end
        %finding mic position based on choice of HRIR
        switch HRIR_subdir
            case 'SCUT'
                mic_position = mic_pos_SCUT;
            case 'ARI_bte'
                mic_position = mic_pos_ARI_bte;
            case 'MMHR-HRTF\BKwHA'
                mic_position = mic_pos_BKwHA_bte;
            case 'FF_KOELN'
                mic_position = mic_pos_FF_KOELN;
        end
        %generating diffuse noise (diffuse and isotropic noise are equivalent in this case, although the two definitions are somewhat different)        
        diffuse_noise = gen_isotropic_noise_2020_12_17(noise_mono,fs_noise,mic_position,size(anechoic_speech,1));
        %cropping size of diffuse noise file
        if size(diffuse_noise,1) ~= size(speech_binaural,1)
            diffuse_noise = diffuse_noise(1:size(speech_binaural,1),1);
        end

        fs = fs_speech;


        %creating noisy speech file with appropriate SNRs. 
        %activlev calculates the SNR of speech based on the ITU-T P.56 standard.
        %This is needed as speech has pauses and therefore traditional SNR
        %calculation is not necessarily representative of true speech
        %level. ITU-T P.56 looks to fix that.
        
        %activelev is performed twice because it's not entirely accurate
        %the first time
        s_db=v_activlev(speech_binaural,16000,'d');  % speech level in dB
        mod_binaural_speech = 10^((-(s_db))/20)*speech_binaural;
        s_db=v_activlev(mod_binaural_speech,16000,'d');  % speech level in dB
        mod_binaural_speech = 10^((-(s_db))/20)*mod_binaural_speech;
        s_db = v_activlev(mod_binaural_speech,16000,'d');  % speech level in dB
        n_db = mean(pow2db(sum(diffuse_noise.^2)/size(diffuse_noise,1)));

        SNR_diff_index = SNR_diff_index_pool(randperm(length(SNR_diff_index_pool),1));
        SNR_diff_index_pool=SNR_diff_index_pool(SNR_diff_index_pool~=SNR_diff_index);
        %noise speech output
        binaural_out = mod_binaural_speech + 10^(-(SNR_diff(SNR_diff_index) + n_db)/20)*diffuse_noise;

        %now comes the mess of generating the directional noise components
        %and adding them to the noisy speech file
        direc_index = direc_pool(randperm(length(direc_pool),1));
        direc_pool =direc_pool(direc_pool~=direc_index);
        %If we have either one or two interfers based on our parameter
        if direc_on(direc_index) == 1 || direc_on(direc_index) == 2
            direc_type_index{1} = direc_type_pool(randperm(length(direc_type_pool),1));
            direc_type_pool = direc_type_pool(direc_type_pool~=direc_type_index{1});
            
            SNR_direc_index{1} = SNR_direc_index_pool(randperm(length(SNR_direc_index_pool),1));
            
            SNR_direc_index_pool=SNR_direc_index_pool(SNR_direc_index_pool~=SNR_direc_index{1});
            
            SNR_direc_index{2} = randi(length(SNR_direc));

            
            



            %if we are simulating RIR (decided previously)
            if ~any(alpha_vals >= 1)
                %we need a new pool for directional azimuth which has all
                %parameters, except for the one already chosen
                if az_index > 1
                    mod_az_index_pool = 1:(az_index - 1);
                    if az_index < length(az_target)
                        mod_az_index_pool = [mod_az_index_pool, (az_index + 1):length(az_target)];
                    end
                else
                    mod_az_index_pool = (az_index + 1):length(az_target);
                end
                %selection from there is random (not pseudorandom)
                az_index_direc = randperm(length(mod_az_index_pool),2);

                %selection of elevation angle is also random (not
                %pseudorandom)
                el_index_direc = randperm(length(el_target),2);
                
                %need to adjust azimuth angle in format for RIR generation
                az_target_rir_gen_direc(1) = az_target(az_index_direc(1)) + 90;
                if az_target_rir_gen_direc(1) > 180
                    az_target_rir_gen_direc(1) = az_target_rir_gen_direc(1) - 360;
                end
                %we now need to choose a radius for the directional
                %interferer, this is done in the following way to maximize
                %coverage of different radii
                clear radii_considered;
                switch HRIR_subdir
                    case 'SCUT'
                        radius_increment = 0.05*max_radius_target;
                        radius_considered_high = radius_target;
                        
                        count_index = 1;
                        while radius_considered_high < 1.1 && radius_considered_high <= 0.75*max_radius_target
                            radii_considered(count_index) = radius_considered_high;
                            count_index = count_index + 1;
                            radius_considered_high = radius_considered_high + radius_increment;
                        end
    
                        radius_considered_low = radius_target;
                        
                        if el_target(el_index_direc(1)) ~= -15 || el_target(el_index_direc(1)) ~= 0 || el_target(el_index_direc(1)) ~= 15
                            dist(1) = abs(el_target(el_index_direc(1)) + 15);
                            dist(2) = abs(el_target(el_index_direc(1)));
                            dist(3) = abs(el_target(el_index_direc(1)) - 15);
                            [~,min_index] = min(dist);
                            if min_index == 1
                                HRIR_el_direc(1) = -15;
                            elseif min_index == 2
                                HRIR_el_direc(1) = 0;
                            elseif min_index == 3
                                HRIR_el_direc(1) = 15;
                            else
                                error('wrong min index');
                            end
                        end
                        while radius_considered_low >= 0.35 && radius_considered_low >= 0.25*max_radius_target
                            if radius_considered_low ~= radius_target
                                radii_considered(count_index) = radius_considered_low;
                                count_index = count_index + 1;
                            end
                            radius_considered_low = radius_considered_low - radius_increment;
                        end
                        considered_index = randi(length(radii_considered));
                        radius_target_direc(1) = radii_considered(considered_index);
                        radius_HRIR_direc = round(radius_target_direc(1),1);
                        if radius_HRIR_direc > 1
                            radius_HRIR_direc = 1;
                        end
                    case 'ARI_bte'
                        radius_increment = 0.05*max_radius_target;
                        radius_considered_high = radius_target;
                        
                        count_index = 1;
                        while radius_considered_high < 1.45 && radius_considered_high <= 0.75*max_radius_target
                            radii_considered(count_index) = radius_considered_high;
                            count_index = count_index + 1;
                            radius_considered_high = radius_considered_high + radius_increment;
                        end
    
                        radius_considered_low = radius_target;
                        while radius_considered_low >= 1.1 && radius_considered_low >= 0.25*max_radius_target
                            if radius_considered_low ~= radius_target
                                radii_considered(count_index) = radius_considered_low;
                                count_index = count_index + 1;
                            end
                            radius_considered_low = radius_considered_low - radius_increment;
                        end
                        considered_index = randi(length(radii_considered));
                        radius_target_direc(1) = radii_considered(considered_index);
                        radius_HRIR_direc = 1.2;
                        HRIR_el_direc(1) = el_target(el_index_direc(1));
                        
                    case 'MMHR-HRTF\BKwHA'
                        radius_increment = 0.05*max_radius_target;
                        radius_considered_high = radius_target;
                        
                        count_index = 1;
                        while radius_considered_high < 2.7 && radius_considered_high <= 0.75*max_radius_target
                            radii_considered(count_index) = radius_considered_high;
                            count_index = count_index + 1;
                            radius_considered_high = radius_considered_high + radius_increment;
                        end
    
                        radius_considered_low = radius_target;
                        while radius_considered_low >= 1.45 && radius_considered_low >= 0.25*max_radius_target
                            if radius_considered_low ~= radius_target
                                radii_considered(count_index) = radius_considered_low;
                                count_index = count_index + 1;
                            end
                            radius_considered_low = radius_considered_low - radius_increment;
                        end
                        considered_index = randi(length(radii_considered));
                        radius_target_direc(1) = radii_considered(considered_index);
                        radius_HRIR_direc = 1.7;
                        HRIR_el_direc(1) = el_target(el_index_direc(1));
                        
                    case 'FF_KOELN'
                        radius_increment = 0.05*max_radius_target;
                        radius_considered_high = radius_target;
                        
                        count_index = 1;
                        while radius_considered_high <= 0.75*max_radius_target
                            radii_considered(count_index) = radius_considered_high;
                            count_index = count_index + 1;
                            radius_considered_high = radius_considered_high + radius_increment;
                        end
    
                        radius_considered_low = radius_target;
                        while radius_considered_low >= 2.7 && radius_considered_low >= 0.25*max_radius_target
                            if radius_considered_low ~= radius_target
                                radii_considered(count_index) = radius_considered_low;
                                count_index = count_index + 1;
                            end
                            radius_considered_low = radius_considered_low - radius_increment;
                        end
                        considered_index = randi(length(radii_considered));
                        radius_target_direc(1) = radii_considered(considered_index);
                        radius_HRIR_direc = 3.25;
                        HRIR_el_direc(1) = el_target(el_index_direc(1));
                end
                %simulating RIR based on parameters
                radius_target_direc_ratio(1) = radius_target_direc(1)/max_radius_target;
                
                [h_omni_direc{1}, radius_target_direc(1),~] = gen_spatial_IR_2021_05_27(az_target_rir_gen_direc(1), el_target(el_index_direc(1)), radius_target_direc_ratio(1), room_dimension, alpha_vals);
                %reading HRIR based on parameters
                HRIR_dir_direc{1} = strcat('az_',num2str(az_target(az_index_direc(1))),'_el_',num2str(HRIR_el_direc(1)),'_radius_',num2str(radius_HRIR_direc(1)));
    
                HRIR_out_dir_direc{1} = strcat(HRIR_base_dir,'\',HRIR_subdir,'\',HRIR_dir_direc{1},'.wav');

                [HRIR_direc{1},fs_HRIR_direc(1)] = audioread(HRIR_out_dir_direc{1});
   
                HRIR_direc{1} = resample(HRIR_direc{1},fs_speech,fs_HRIR_direc(1));
 
                BRIR_direc{1} = fftfilt(HRIR_direc{1},h_omni_direc{1});
                %now in the case of two interferers we need to repeat this
                %process
                if direc_on(direc_index) == 2
                    direc_type_index{2} = randi(2);
                    az_target_rir_gen_direc(2) = az_target(az_index_direc(2)) + 90;
                    if az_target_rir_gen_direc(2) > 180
                        az_target_rir_gen_direc(2) = az_target_rir_gen_direc(2) - 360;
                    end
                
                    considered_index = randi(length(radii_considered));
                    radius_target_direc(2) = radii_considered(considered_index);
                    radius_target_direc_ratio(2) = radius_target_direc(2)/max_radius_target;
                    
                    switch HRIR_subdir
                        case 'SCUT'
                            radius_increment = 0.05*max_radius_target;
                            radius_considered_high = radius_target;

                            count_index = 1;
                            while radius_considered_high < 1.1 && radius_considered_high <= 0.75*max_radius_target
                                radii_considered(count_index) = radius_considered_high;
                                count_index = count_index + 1;
                                radius_considered_high = radius_considered_high + radius_increment;
                            end

                            radius_considered_low = radius_target;

                            if el_target(el_index_direc(2)) ~= -15 || el_target(el_index_direc(2)) ~= 0 || el_target(el_index_direc(2)) ~= 15
                                dist(1) = abs(el_target(el_index_direc(2)) + 15);
                                dist(2) = abs(el_target(el_index_direc(2)));
                                dist(3) = abs(el_target(el_index_direc(2)) - 15);
                                [~,min_index] = min(dist);
                                if min_index == 1
                                    HRIR_el_direc(2) = -15;
                                elseif min_index == 2
                                    HRIR_el_direc(2) = 0;
                                elseif min_index == 3
                                    HRIR_el_direc(2) = 15;
                                else
                                    error('wrong min index');
                                end
                            end
                        otherwise
                            HRIR_el_direc(2) = el_target(el_index_direc(2));
                    end
                    
                    [h_omni_direc{2}, radius_target_direc(2),~] = gen_spatial_IR_2021_05_27(az_target_rir_gen_direc(2), el_target(el_index_direc(2)), radius_target_direc_ratio(2), room_dimension, alpha_vals);
                
                    HRIR_dir_direc{2} = strcat('az_',num2str(az_target(az_index_direc(2))),'_el_',num2str(HRIR_el_direc(2)),'_radius_',num2str(radius_HRIR_direc));

                    HRIR_out_dir_direc{2} = strcat(HRIR_base_dir,'\',HRIR_subdir,'\',HRIR_dir_direc{2},'.wav');

                    [HRIR_direc{2},fs_HRIR_direc(2)] = audioread(HRIR_out_dir_direc{2});

                    HRIR_direc{2} = resample(HRIR_direc{2},fs_speech,fs_HRIR_direc(2));

                    BRIR_direc{2} = fftfilt(HRIR_direc{2},h_omni_direc{2});
                end
                
            else
                %this is what happens when we did not simulate RIR for
                %primary target

                HRIR_subdir_direc_dir = strcat(HRIR_base_dir,'\',HRIR_subdir);
                file_type = '.wav';
                wav_files = dir(strcat(HRIR_subdir_dir,'\','*',file_type));
                wav_files = {wav_files.name};
                nb_files = length(wav_files);
                
                rand_file_ind = randperm(nb_files,2);

                HRIR_out_dir_direc{1} = strcat(HRIR_subdir_direc_dir,'\',wav_files{rand_file_ind(1)});
                [HRIR_direc{1},fs_HRIR_direc] = audioread(HRIR_out_dir_direc{1});

                HRIR_direc{1} = resample(HRIR_direc{1},fs_speech,fs_HRIR_direc);

                BRIR_direc{1} = HRIR_direc{1};
                if direc_on(direc_index) == 2
                    HRIR_out_dir_direc{2} = strcat(HRIR_subdir_direc_dir,'\',wav_files{rand_file_ind(2)});
                    [HRIR_direc{2},fs_HRIR_direc] = audioread(HRIR_out_dir_direc{2});

                    HRIR_direc{2} = resample(HRIR_direc{2},fs_speech,fs_HRIR_direc);

                    BRIR_direc{2} = HRIR_direc{2};
                end
            end
                
            %in order to limit the effect of the reverberation already
            %present in the noise files used as directional sources to
            %interfere with the reverberation calculated in the RIR, as
            %well as to have a meaningful directional interferer, the
            %loudest part of the noise file is chosen to serve as the
            %directional noise file (louder means higher direct to
            %reverberant ratio). This is done in the following way
            if direc_type(direc_type_index{1}) == 0
                direc_file_index = randi(size(noise_wav_dirs,1));
                noise_direc_dir{1} = noise_wav_dirs{direc_file_index};
                [direc_full, fs_direc] = audioread(noise_direc_dir{1});
                if size(direc_full,2) > 1
                    direc_full = mean(direc_full,2);
                end
                if fs_direc ~= fs
                    direc_full = resample(direc_full,fs,fs_direc);
                end
                [~,I] = max(direc_full);
                overshoot = round(size(speech_binaural,1)/2);
                out_length = size(speech_binaural,1);
                if I >= out_length/2
                    while overshoot + I > length(direc_full) || I - (out_length - overshoot) + 1 < 1
                        overshoot = round(overshoot/2);
                        if overshoot < 5
                            overshoot = 0;
                        end
                    end
                    direc = direc_full((I+overshoot - out_length + 1):(I + overshoot));

                elseif I <= out_length/2
                    while I - overshoot < 1 || I + (out_length - overshoot -1) > length(direc_full)
                        overshoot = round(overshoot/2);
                        if overshoot < 5
                            overshoot = 0;
                        end
                    end
                    direc = direc_full((I - overshoot):(I-overshoot + out_length - 1));
                end
                %generating spatial and reverberant directional interfer
                direc_binaural{1} = fftfilt(BRIR_direc{1},direc);

                nd_db = mean(pow2db(sum(direc_binaural{1}.^2)/size(direc_binaural{1},1)));
                direc_binaural{1} = 10^((-(nd_db) - SNR_direc(SNR_direc_index{1}))/20)*direc_binaural{1};
            %when the interfer is speech we don't need to worry about
            %finding the loudest part of the noise file, so the process is
            %simpler
            elseif direc_type(direc_type_index{1}) == 1
                TIMIT_direc_dir{1} = randsample(TIMIT_wav_dirs,1);
                [direc,fs_direc] = audioread(TIMIT_direc_dir{1}{1});
                if fs_direc ~= fs
                    direc = resample(direc,fs,fs_direc);
                end
                if length(direc) > size(speech_binaural,1)
                    direc = direc(1:size(speech_binaural,1),1);
                elseif length(direc) < size(speech_binaural,1)
                    direc = [direc;zeros(size(speech_binaural,1) - length(direc),1)];
                end
                direc_binaural{1} = fftfilt(BRIR_direc{1},direc);
                sd_db = v_activlev(direc_binaural{1},16000,'d');
                direc_binaural{1} = 10^((-(sd_db) - SNR_direc(SNR_direc_index{1}))/20)*direc_binaural{1};
            end
            %in any case at this point we will have a spatialized
            %directional interfer at the appropriate SNR, which we can add
            %to the already made noisy speech output.
            binaural_out = binaural_out + direc_binaural{1};
        end
        %if we have two directional interferers we need to repeat the
        %process for generating the finalised spatialized directional
        %output again for the second interferer
        if direc_on(direc_index) == 2
            direc_type_index{2} = randi(2);

            if direc_type(direc_type_index{2}) == 0
                direc_file_index = randi(size(noise_wav_dirs,1));
                noise_direc_dir{2} = noise_wav_dirs{direc_file_index};
                [direc_full, fs_direc] = audioread(noise_direc_dir{2});
                if size(direc_full,2) > 1
                    direc_full = mean(direc_full,2);
                end
                if fs_direc ~= fs
                    direc_full = resample(direc_full,fs,fs_direc);
                end
                [~,I] = max(direc_full);
                overshoot = round(size(speech_binaural,1)/2);
                out_length = size(speech_binaural,1);
                if I >= out_length/2
                    while overshoot + I > length(direc_full) || I - (out_length - overshoot) + 1 < 1
                        overshoot = round(overshoot/2);
                        if overshoot < 5
                            overshoot = 0;
                        end
                    end
                    direc = direc_full((I+overshoot - out_length + 1):(I + overshoot));

                elseif I <= out_length/2
                    while I - overshoot < 1 || I + (out_length - overshoot -1) > length(direc_full)
                        overshoot = round(overshoot/2);
                        if overshoot < 5
                            overshoot = 0;
                        end
                    end
                    direc = direc_full((I - overshoot):(I-overshoot + out_length - 1));
                end

                direc_binaural{2} = fftfilt(BRIR_direc{2},direc);
                
                nd_db = mean(pow2db(sum(direc_binaural{2}.^2)/size(direc_binaural{2},1)));
                direc_binaural{2} = 10^((-(nd_db) - SNR_direc(SNR_direc_index{2}))/20)*direc_binaural{2};
            elseif direc_type(direc_type_index{2}) == 1
                TIMIT_direc_dir{2} = randsample(TIMIT_wav_dirs,1);
                [direc,fs_direc] = audioread(TIMIT_direc_dir{2}{1});
                if fs_direc ~= fs
                    direc = resample(direc,fs,fs_direc);
                end
                if length(direc) > size(speech_binaural,1)
                    direc = direc(1:size(speech_binaural,1),1);
                elseif length(direc) < size(speech_binaural,1)
                    direc = [direc;zeros(size(speech_binaural,1) - length(direc),1)];
                end
                direc_binaural{2} = fftfilt(BRIR_direc{2},direc);
                sd_db = v_activlev(direc_binaural{2},16000,'d');
                direc_binaural{2} = 10^((-(sd_db) - SNR_direc(SNR_direc_index{2}))/20)*direc_binaural{2};
            end

            binaural_out = binaural_out + direc_binaural{2};
        end
        %at this point we have the binaural noisy speech output finalized
        
        %we generate the anechoic spatialized speech (for comparison
        %purposes). The final implementation uses the raw TIMIT anechoic
        %speech as clean speech
        anechoic_speech_HRIR = fftfilt(HRIR,anechoic_speech);
        
        %we normalize the output to zero-mean, unit-variance.
        binaural_out = reshape(zscore(binaural_out(:)),size(binaural_out,1),size(binaural_out,2));
        anechoic_speech = reshape(zscore(anechoic_speech(:)),size(anechoic_speech,1),size(anechoic_speech,2));
        anechoic_speech_HRIR = reshape(zscore(anechoic_speech_HRIR(:)),size(anechoic_speech_HRIR,1),size(anechoic_speech_HRIR,2));
            
        %preparation for MBSTOI calculation
        xl = anechoic_speech;
        xr = anechoic_speech;
        xl_HRIR = anechoic_speech_HRIR(:,1);
        xr_HRIR = anechoic_speech_HRIR(:,2);

        yl = binaural_out(:,1);
        yr = binaural_out(:,2);
        %calculate both mbstoi for clean and HRIR anechoic speech
        
        %calculating MBSTOI
        mbstoi = mbstoi_intermediate(xl,xr,yl,yr,fs_speech);
        mbstoi_HRIR = mbstoi_intermediate(xl_HRIR, xr_HRIR, yl, yr, fs_speech);
        
        %doing all the work of creating the directories and saving files
        out_dir = strcat(out_base_dir,'\',num2str(file_counter));
        disp(out_dir);
        mkdir(out_dir);
        mbstoi_file_name = strcat(out_dir,'\','mbstoi.mat');
        mbstoi_HRIR_file_name = strcat(out_dir,'\','mbstoi_hrtf.mat');
        save(mbstoi_file_name,'mbstoi');
        save(mbstoi_HRIR_file_name, 'mbstoi_HRIR');
        %save(mat_file_name,'mbstoi_norm','mbstoi','equal_check');

        out_clean_wav_dir = strcat(out_dir,'\','clean.wav');
        out_clean_HRIR_wav_dir = strcat(out_dir,'\','clean_hrtf.wav');
        out_mixed_wav_dir = strcat(out_dir,'\','mixed.wav');
        v_writewav(binaural_out,16000,out_mixed_wav_dir,[],[],[],[]);
        v_writewav(anechoic_speech,16000,out_clean_wav_dir,[],[],[],[]);
        v_writewav(anechoic_speech_HRIR,16000,out_clean_HRIR_wav_dir,[],[],[],[]);
        
        %saving log file so that we can see all of the settings when
        %referencing audio files after the fact
        fileID = fopen(strcat(out_dir,'\','log.txt'),'w');
        fprintf(fileID,'mbstoi.mat contains the full mbstoi array, where the first dimension corresponds to the one-third octave bands and the second dimension correpsonds to the intermediate intelligbility indexes (blocks of 4096 samples at 10000 Khz of pure speech');
        fprintf(fileID,'The clean reference speech used was anechoic TIMIT speech \n');
        
        fprintf(fileID,'mbstoi_hrtf.mat contains the full mbstoi array, using the clean_hrf speech as the clean reference speech, where the first dimension corresponds to the one-third octave bands and the second dimension correpsonds to the intermediate intelligbility indexes (blocks of 4096 samples at 10000 Khz of pure speech');
        fprintf(fileID,'The clean_hrtf reference speech is anechoic TIMIT speech convolved with the HRIR described below\n');
        fprintf(fileID,'\n\n');
        fprintf(fileID,strcat(strrep(strcat('directory of TIMIT file used: ',TIMIT_wav_dirs{TIMIT_wav_index,1}),'\','/'),' \n'));
        fprintf(fileID,'SNR diffuse noise: %.1f \n',SNR_diff(SNR_diff_index));
        fprintf(fileID,strcat(strrep(strcat('directory of noise file used for diffuse noise: ',noise_dir),'\','/'),' \n'));
        
        fprintf(fileID,'\n\n');
        if ~any(alpha_vals >= 1)
            fprintf(fileID,'Impulse Response Parameters: \n');
            fprintf(fileID,'Simulated \n');
            fprintf(fileID,'room dimensions [%.1f,%.1f,%.1f] [x,y,z] \n',room_dimension);
            if ~any(alpha_vals >= 1)
                fprintf(fileID,'radius of target (m): %.2f \n',radius_target);
            end
            fprintf(fileID,'azimuth angle of target (degrees): %.2f \n',az_target(az_index));
            fprintf(fileID,'0 degrees azimuth is the direction facing the listener, with 90 degrees being pure left channel and 270 degrees being pure right channel \n');

            fprintf(fileID,'elevation angle of target (degrees): %.2f \n',el_target(el_index));

            fprintf(fileID,'reverberation parameter alpha: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f,] [alpha_{x1}, alpha_{x2}, alpha_{y1}, alpha_{y2}, alpha_{z1}, alpha_{z2}] \n',alpha_array(alpha_index,:) + alpha_add(1,alpha_add_index));
            fprintf(fileID,strcat(strrep(strcat('directory of HRIR file used: ',HRIR_out_dir),'\','/'),' \n'));
        else
            fprintf(fileID,'anechoic \n');
            fprintf(fileID,strcat(strrep(strcat('directory of HRIR file used: ',HRIR_out_dir),'\','/'),' \n'));
        end
        if direc_on(direc_index) == 1 || direc_on(direc_index) == 2
            fprintf(fileID,'Directional Noise Included \n');
            if direc_on(direc_index) == 1
                fprintf(fileID,'Number of directional sources: 1 \n');
            else
                fprintf(fileID,'Number of directional sources: 2 \n');
            end
            fprintf(fileID,'First directional source: \n');
            fprintf(fileID,'SNR directional noise: %.1f \n',SNR_direc(SNR_direc_index{1}));
            if ~any(alpha_vals >= 1)
                fprintf(fileID,'reverberant \n');
                fprintf(fileID,'azimuth angle of direc. target (degrees): %.2f \n',az_target(az_index_direc(1)));
                fprintf(fileID,'elevation angle of direc. target (degrees): %.2f \n',el_target(el_index_direc(1)));
                fprintf(fileID,'radius of direc. target (m): %.2f \n',radius_target_direc(1));
                fprintf(fileID,strcat(strrep(strcat('directory of HRIR used: ',HRIR_out_dir_direc{1}),'\','/'),' \n'));
                
            else
                fprintf(fileID,'anechoic \n');
                fprintf(fileID,strcat(strrep(strcat('directory of HRIR used: ',HRIR_out_dir_direc{1}),'\','/'),' \n'));
                
            end
            if direc_type(direc_type_index{1}) == 0
                fprintf(fileID,'Non-speech directional noise: \n');
                fprintf(fileID,strcat(strrep(strcat('directory of noise file used: ',noise_direc_dir{1}),'\','/'),' \n'));
            elseif direc_type(direc_type_index{1}) == 1
                fprintf(fileID,'Speech directional noise: \n');
                fprintf(fileID,strcat(strrep(strcat('directory of speech file used: ',TIMIT_direc_dir{1}{1}),'\','/'),' \n'));
            end
            if direc_on(direc_index) == 2
                fprintf(fileID,'Second directional source: \n');
                fprintf(fileID,'SNR directional noise: %.1f \n',SNR_direc(SNR_direc_index{2}));
                if ~any(alpha_vals >= 1)
                    fprintf(fileID,'reverberant \n');
                    fprintf(fileID,'azimuth angle of direc. target (degrees): %.2f \n',az_target(az_index_direc(2)));
                    fprintf(fileID,'elevation angle of direc. target (degrees): %.2f \n',el_target(el_index_direc(2)));
                    fprintf(fileID,'radius of direc. target (m): %.2f \n',radius_target_direc(2));
                    fprintf(fileID,strcat(strrep(strcat('directory of HRIR used: ',HRIR_out_dir_direc{2}),'\','/'),' \n'));

                else
                    fprintf(fileID,'anechoic \n');
                    fprintf(fileID,strcat(strrep(strcat('directory of HRIR used: ',HRIR_out_dir_direc{2}),'\','/'),' \n'));

                end
                if direc_type(direc_type_index{2}) == 0
                    fprintf(fileID,'Non-speech directional noise: \n');
                    fprintf(fileID,strcat(strrep(strcat('directory of noise file used: ',noise_direc_dir{2}),'\','/'),' \n'));
                elseif direc_type(direc_type_index{2}) == 1
                    fprintf(fileID,'Speech directional noise: \n');
                    fprintf(fileID,strcat(strrep(strcat('directory of speech file used: ',TIMIT_direc_dir{2}{1}),'\','/'),' \n'));
                end
            end
        end

        fclose(fileID);
        
        file_counter = file_counter + 1;


    end
end
    
    
