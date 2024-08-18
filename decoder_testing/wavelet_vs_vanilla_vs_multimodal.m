clear
addpath("/home/cyberspace007/mpicek/master_project/decoder_testing/ModelRecomputation")
load /home/cyberspace007/mpicek/master_project/decoder_testing/NRColors.mat
NRColors.VoltageYellow = [235, 227, 4] / 255;
NRColors.DeepViolet = [108, 62, 144] / 255;
NRColors.Magenta = [211, 53, 190] / 255;
NRColors.FadeOrange = [254, 158, 83] / 255;
NRColors.DeepOrange = [237, 113, 26] / 255;
NRColors.SpaceGreen = [85, 156, 78] / 255;
NRColors.LightningGreen = [126, 232, 115] / 255;
NRColors.DarkRed = [163, 27, 33] / 255;
NRColors.DeepRed = [201, 23, 44] / 255;
NRColors.DeepMagenta = [179, 0, 118] / 255;
NRColors.DeepBlue = [4, 126, 166] / 255;
NRColors.DeepCyan = [4, 127, 109] / 255;

nr_red_white_map = [linspace(1, NRColors.ImpulseRed(1));linspace(1, NRColors.ImpulseRed(2));linspace(1, NRColors.ImpulseRed(3))]';

function f1_scores = compute_f1_score(confusion_matrix)
    % Compute the F1 Score for each class given a confusion matrix.
    % Parameters:
    % confusion_matrix (matrix): A square matrix of size (n_classes, n_classes) where
    %                            confusion_matrix(i, j) is the count of true class i
    %                            predicted as class j.
    % Returns:
    % f1_scores (vector): A vector containing the F1 Score for each class.
    
    % Number of classes
    confusion_matrix = confusion_matrix';
    n_classes = size(confusion_matrix, 1);
    
    % Initialize vectors to hold precision, recall, and F1 Score for each class
    precision = zeros(n_classes, 1);
    recall = zeros(n_classes, 1);
    f1_scores = zeros(n_classes, 1);
    
    % Calculate precision, recall, and F1 Score for each class
    for i = 1:n_classes
        tp = confusion_matrix(i, i); % True positives
        fp = sum(confusion_matrix(:, i)) - tp; % False positives
        fprintf("tp: %d\n", tp);
        fprintf("fp: %d\n", fp);
        fn = sum(confusion_matrix(i, :)) - tp; % False negatives
        fprintf("fn: %d\n", fn);
        
        % Precision calculation (handle division by zero)
        if (tp + fp) > 0
            precision(i) = tp / (tp + fp);
        else
            precision(i) = 0;
        end
        
        % Recall calculation (handle division by zero)
        if (tp + fn) > 0
            recall(i) = tp / (tp + fn);
        else
            recall(i) = 0;
        end
        
        % F1 Score calculation (handle division by zero)
        if (precision(i) + recall(i)) > 0
            f1_scores(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        else
            f1_scores(i) = 0;
        end
    end
end

datapath_session_1 = '/media/cyberspace007/T7/martin/decoder_testing/processed/day_9/rec2';
datapath_session_2 = '/media/cyberspace007/T7/martin/decoder_testing/processed/day_9/rec4';
datapath_session_4 = '/media/cyberspace007/T7/martin/decoder_testing/processed/day_9/rec6';

% loading files generated from .dt5
session1 = load([datapath_session_1, '/all.mat']);
session2 = load([datapath_session_2, '/all.mat']);
session4 = load([datapath_session_4, '/all.mat']);

% vanilla BrainGPT is the good old BrainGPT without any multimodality
% modification
xLatent1_vanilla = load("/media/cyberspace007/T7/martin/decoder_testing/processed/day_9/rec2/latent_features_vanilla.mat").xLatent';
xLatent2_vanilla = load("/media/cyberspace007/T7/martin/decoder_testing/processed/day_9/rec4/latent_features_vanilla.mat").xLatent';
xLatent4_vanilla = load("/media/cyberspace007/T7/martin/decoder_testing/processed/day_9/rec6/latent_features_vanilla.mat").xLatent';

% these latents are from the multimodal BrainGPT
xLatent1 = load("/media/cyberspace007/T7/martin/decoder_testing/processed/day_9/rec2/latent_features_DINO_SSL_full_retrain_29.mat").xLatent';
xLatent2 = load("/media/cyberspace007/T7/martin/decoder_testing/processed/day_9/rec4/latent_features_DINO_SSL_full_retrain_29.mat").xLatent';
xLatent4 = load("/media/cyberspace007/T7/martin/decoder_testing/processed/day_9/rec6/latent_features_DINO_SSL_full_retrain_29.mat").xLatent';

yDesired = [session1.yDesired, session2.yDesired, session4.yDesired];
IsUpdating = [session1.IsUpdating, session2.IsUpdating, session4.isUpdating];
probaPredicted = [session1.probaPredicted, session2.probaPredicted, session4.probaPredicted];

xLatent_multimodal = [xLatent1; xLatent2; xLatent4];
xLatent_vanilla = [xLatent1_vanilla; xLatent2_vanilla; xLatent4_vanilla];

x = [reshape(session1.x, size(session1.x,1), []); reshape(session2.x, size(session2.x,1), []); reshape(session4.X, [], size(session4.X,4))']; % flattened x

%%
% VISUALIZING THE WHOLE SESSION
size(xLatent1)
size(xLatent2)
size(xLatent4)
size(x)
size(xLatent_vanilla)
size(xLatent_multimodal)

subplot(2, 1, 2);
plot(IsUpdating);
subplot(2, 1, 1);
plot(yDesired([1,2,9,10,12,13], :)');
legend(arrayfun(@(x) sprintf('Category %d', x), 1:size(yDesired, 1), 'UniformOutput', false));

%%
% TRYING DIFFERENT SIZES OF THE TRAINING DATASET
rng(1);
data_portions = 1.0:0.1:1.0;

selected_states = [1,2,9,10,12,13];
state_indices = [1,2,9,10,12,13];
selected_range = [18190, 20310];
remove_range = [19850, 20040]; % relative to selected_range(1)
shift_size = [-2, 0];
n_states = size(selected_states,2);

f1_scores_multimodal = zeros(size(data_portions));
f1_scores_vanilla = zeros(size(data_portions));
f1_scores_wavelet = zeros(size(data_portions));

accuracies_multimodal = zeros(size(data_portions));
stds_multimodal = zeros(size(data_portions));
accuracies_all_multimodal = zeros([size(data_portions), 5]);

accuracies_wavelet = zeros(size(data_portions));
stds_wavelet = zeros(size(data_portions));
accuracies_all_wavelet = zeros([size(data_portions), 5]);

accuracies_vanilla = zeros(size(data_portions));
stds_vanilla = zeros(size(data_portions));
accuracies_all_vanilla = zeros([size(data_portions), 5]);

all_confusion_matrices_wavelet = zeros(length(data_portions), 6, 6);
all_confusion_matrices_vanilla = zeros(length(data_portions), 6, 6);
all_confusion_matrices_multimodal = zeros(length(data_portions), 6, 6);

for portion_idx = 1:length(data_portions)
    portion = data_portions(portion_idx);
    
    newUpdate = zeros(size(IsUpdating));

    % Extract indices of the states of interest
    state_indices = [1,2,9,10,12,13];

    % Loop through each state and update newUpdate to include only half of each class
    for state = 1:length(state_indices)
        current_state = state_indices(state);

        % Find the indices of the current state in yDesired
        state_pos = find(yDesired(current_state, :) & IsUpdating == 1);

        % Randomly shuffle the indices
        state_pos = state_pos(randperm(length(state_pos)));

        selected_positions = state_pos(1:ceil(length(state_pos) * portion));

        % Update newUpdate to include the selected indices
        newUpdate(selected_positions) = 1;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % THIS PART OF CODE HANDLES A VERY WEIRD CONDITION - WHEN IS_UPDATING
    % == 1 AND ALL OF THE CLASSES ARE == 0 (SO NO STATE IS PRESENT, NOT EVEN THE RESTING STATE)
    % Find indices where all rows in yDesired are zero and IsUpdating is 1
    zero_indices = all(yDesired == 0, 1) & IsUpdating == 1;
    zero_indices_count = sum(zero_indices);
    % Randomly shuffle the zero indices
    zero_indices = find(zero_indices);
    zero_indices = zero_indices(randperm(length(zero_indices)));
    % Select the portion of the zero indices
    selected_zero_positions = zero_indices(1:ceil(length(zero_indices) * portion));
    newUpdate(selected_zero_positions) = 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    fprintf("Portion %d/%d = %d:\n", portion_idx, length(data_portions), portion);
    % Print the new counts for each state to verify
    new_state_counts = sum(yDesired(state_indices, newUpdate == 1), 2);
    for state = 1:length(new_state_counts)
        fprintf('State %d occurred %d times in the new update\n', state_indices(state), new_state_counts(state));
    end

    x_train = x(newUpdate==1,:);
    z_train_multimodal = xLatent_multimodal(newUpdate==1,:);
    z_train_vanilla = xLatent_vanilla(newUpdate==1,:);
    u_train = yDesired(:,newUpdate==1)';

    shift_size = [-2, 0];
    Data_wavelet.X = x_train;
    Data_wavelet.Y = circshift(u_train, shift_size);

    Data_latent_multimodal.X = z_train_multimodal;
    Data_latent_multimodal.Y = circshift(u_train, shift_size);

    Data_latent_vanilla.X = z_train_vanilla;
    Data_latent_vanilla.Y = circshift(u_train, shift_size);

    Parameters_multimodal.SizeX = [1 1 size(z_train_multimodal,2)];
    [fullModel_multimodal,PredictionBeta_multimodal,PredictionBias_multimodal] = fastREWKernelNPLS(Data_latent_multimodal,[],Parameters_multimodal);

    Parameters.SizeX = [1 1 size(x_train,2)];
    [fullModel,PredictionBeta,PredictionBias] = fastREWKernelNPLS(Data_wavelet,[],Parameters);

    Parameters_vanilla.SizeX = [1 1 size(z_train_vanilla,2)];
    [fullModel_vanilla,PredictionBeta_vanilla,PredictionBias_vanilla] = fastREWKernelNPLS(Data_latent_vanilla,[],Parameters_vanilla);

    x_test = x(IsUpdating==0,:);
    z_test_multimodal = xLatent_multimodal(IsUpdating==0,:);
    z_test_vanilla = xLatent_vanilla(IsUpdating==0,:);
    u_test = yDesired(:,IsUpdating==0)';

    transition_train = transitions(u_train);
    norm_transition = transition_train ./ sum(transition_train);
    norm_transition(isnan(norm_transition)) = 0;

    pred_wavelet = getGatePrediction(x_test, PredictionBeta, PredictionBias, norm_transition, []);
    pred_multimodal = getGatePrediction(z_test_multimodal, PredictionBeta_multimodal, PredictionBias_multimodal, norm_transition, []);
    pred_vanilla = getGatePrediction(z_test_vanilla, PredictionBeta_vanilla, PredictionBias_vanilla, norm_transition, []);

    selected_states = [1,2,9,10,12,13];
    n_states = size(selected_states,2);
    selected_range = [18190, 20310];
    % remove from 1660 to 1850
    
    remove_range = [19850, 20040]; % relative to selected_range(1)
    sel_pred_wavelet = [pred_wavelet(selected_range(1):remove_range(1),selected_states); pred_wavelet(remove_range(2):selected_range(2),selected_states)];
    sel_pred_multimodal = [pred_multimodal(selected_range(1):remove_range(1),selected_states); pred_multimodal(remove_range(2):selected_range(2),selected_states)];
    sel_pred_vanilla = [pred_vanilla(selected_range(1):remove_range(1),selected_states); pred_vanilla(remove_range(2):selected_range(2),selected_states)];
    sel_u_test = [u_test(selected_range(1):remove_range(1),selected_states); u_test(remove_range(2):selected_range(2),selected_states)];
    sel_xLatent_multimodal = [z_test_multimodal(selected_range(1):remove_range(1),:); z_test_multimodal(remove_range(2):selected_range(2),:)];
    sel_xLatent_vanilla = [z_test_vanilla(selected_range(1):remove_range(1),:); z_test_vanilla(remove_range(2):selected_range(2),:)];
    sel_x = [x_test(selected_range(1):remove_range(1),:); x_test(remove_range(2):selected_range(2),:)];
    sub_sel_probas = probaPredicted(:,IsUpdating==0);
    sel_online_proba = [sub_sel_probas(selected_states,selected_range(1):remove_range(1))'; sub_sel_probas(selected_states,remove_range(2):selected_range(2))'];


    % compute  delay per class
    delays = [];
    for i = 2:n_states
        % Find indices where cue is active
        [del,lags] = xcorr(sel_pred_wavelet(:,i), sel_u_test(:,i));
        [~,id] = max(del);
        delays = [delays lags(id)];
        lags(id);
    end
    delay_wavelet = sum(delays)/length(delays);
    shift_size = [round(delay_wavelet), 0];
    aligned_sel_u_test_wavelet = circshift(sel_u_test, shift_size);


    delays = [];
    for i = 2:n_states
        % Find indices where cue is active
        [del,lags] = xcorr(sel_pred_multimodal(:,i), sel_u_test(:,i));
        [~,id] = max(del);
        delays = [delays lags(id)];
        lags(id);
    end
    delay_multimodal = sum(delays)/length(delays);
    shift_size = [round(delay_multimodal), 0];
    aligned_sel_u_test_multimodal = circshift(sel_u_test, shift_size);


    delays = [];
    for i = 2:n_states
        % Find indices where cue is active
        [del,lags] = xcorr(sel_pred_vanilla(:,i), sel_u_test(:,i));
        [~,id] = max(del);
        delays = [delays lags(id)];
        lags(id);
    end
    delay_vanilla = sum(delays)/length(delays);
    shift_size = [round(delay_vanilla), 0];
    aligned_sel_u_test_vanilla = circshift(sel_u_test, shift_size);
    
    thresh_pred_wavelet = (sel_pred_wavelet==max(sel_pred_wavelet, [], 2));
    acc_wavelet=thresh_pred_wavelet'*aligned_sel_u_test_wavelet;
    acc_wavelet=100*acc_wavelet./sum(aligned_sel_u_test_wavelet);

    thresh_pred_multimodal = (sel_pred_multimodal==max(sel_pred_multimodal, [], 2));
    acc_multimodal=thresh_pred_multimodal'*aligned_sel_u_test_multimodal;
    acc_multimodal=100*acc_multimodal./sum(aligned_sel_u_test_multimodal);

    thresh_pred_vanilla = (sel_pred_vanilla==max(sel_pred_vanilla, [], 2));
    acc_vanilla=thresh_pred_vanilla'*aligned_sel_u_test_vanilla;
    acc_vanilla=100*acc_vanilla./sum(aligned_sel_u_test_vanilla);
    
    wavelet_acc = sum(diag(acc_wavelet(2:6, 2:6))) / 5;
    multimodal_acc = sum(diag(acc_multimodal(2:6, 2:6))) / 5;
    vanilla_acc = sum(diag(acc_vanilla(2:6, 2:6))) / 5;
    
    
    accuracies_vanilla(portion_idx) = vanilla_acc;
    stds_vanilla(portion_idx) = std(diag(acc_vanilla(2:6, 2:6)));      % Replace with actual std values
    accuracies_all_vanilla(1, portion_idx, :) = diag(acc_vanilla(2:6, 2:6));
    
    accuracies_multimodal(portion_idx) = multimodal_acc;
    stds_multimodal(portion_idx) = std(diag(acc_multimodal(2:6, 2:6)));      % Replace with actual std values
    accuracies_all_multimodal(1, portion_idx, :) = diag(acc_multimodal(2:6, 2:6));

    accuracies_wavelet(portion_idx) = wavelet_acc;
    stds_wavelet(portion_idx) = std(diag(acc_wavelet(2:6, 2:6)));   % Replace with actual std values
    accuracies_all_wavelet(1, portion_idx, :) = diag(acc_wavelet(2:6, 2:6));

    acc_multimodal=thresh_pred_multimodal'*aligned_sel_u_test_multimodal;
    acc_vanilla=thresh_pred_vanilla'*aligned_sel_u_test_vanilla;
    acc_wavelet=thresh_pred_wavelet'*aligned_sel_u_test_wavelet;

    f1_scores_multimodal(portion_idx) = mean(compute_f1_score(acc_multimodal));
    f1_scores_vanilla(portion_idx) = mean(compute_f1_score(acc_vanilla));
    f1_scores_wavelet(portion_idx) = mean(compute_f1_score(acc_wavelet));

    all_confusion_matrices_wavelet(portion_idx, :, :) = acc_wavelet;
    all_confusion_matrices_vanilla(portion_idx, :, :) = acc_vanilla;
    all_confusion_matrices_multimodal(portion_idx, :, :) = acc_multimodal;

    disp("multimodal:");
    disp(f1_scores_multimodal);
    disp("vanilla");
    disp(f1_scores_vanilla);
    disp("wavelet");
    disp(f1_scores_wavelet);
    
end

%%
% PLOTTING THE PERFORMANCE ON DIFFERENT SIZES OF THE TRAINING DATASET (F1 SCORES)
colors = [...
    0.812, 0.816, 0.816; ...
    0.298, 0.008, 0.071;...
    0.969, 0.725, 0.690;...
    0.980, 0.322, 0.357;...
    0.788, 0.090, 0.173;...
    0.537, 0.027, 0.118;...
    0.298, 0.008, 0.071];

figure('Color', 'w', 'Position', [100, 100, 1000, 600]);
plot(data_portions*100, f1_scores_multimodal, '-o', 'DisplayName', 'Decoder trained on Multimodal BrainGPT features', 'Color', colors(5, :)); hold on;
plot(data_portions*100, f1_scores_vanilla, '-o', 'DisplayName', 'Decoder trained on BrainGPT features', 'Color', colors(3, :)); hold on;
plot(data_portions*100, f1_scores_wavelet, '-o', 'DisplayName', 'Decoder trained on Wavelet features', 'Color', colors(1, :));
xlabel('Percentage of the training dataset the models were trained on');
ylabel('F1 scores');
title('F1 scores trained decoders on different features with different amounts of training data');
legend;
ylim([0 1]);

% name_png = "F1_SCORES_DINO_SSL_full_retrain_29.png";
% name_svg = "F1_SCORES_DINO_SSL_full_retrain_29.svg";
% 
% directory = '/home/cyberspace007/mpicek/master_project/decoder_testing/figures/';
% 
% full_path_png = fullfile(directory, name_png);
% full_path_svg = fullfile(directory, name_svg);
% 
% exportgraphics(gcf, full_path_png, 'Resolution', 300); % 300 DPI resolution for high quality
% print(full_path_svg, '-dsvg', '-r300', '-opengl');

%%

% COMPARISON OF MODELS: PLOTTING THE PERFORMANCE ON DIFFERENT SIZES OF THE TRAINING DATASET (F1 SCORES)
colors = [...
    0.812, 0.816, 0.816; ...
    0.298, 0.008, 0.071;...
    0.969, 0.725, 0.690;...
    0.980, 0.322, 0.357;...
    0.788, 0.090, 0.173;...
    0.537, 0.027, 0.118;...
    0.298, 0.008, 0.071];

finetuned = [0.5504,    0.7024   , 0.7113   , 0.6615   , 0.7765 ,   0.7706    ,0.7556    ,0.7407   , 0.7364  ,  0.7757];
vanilla = [0.4273,    0.7128  ,  0.6430   , 0.6759  ,  0.7048  ,  0.7345 ,   0.7755  ,  0.6905  ,  0.6889  ,  0.7824];
wavelet = [0.4127 ,   0.2526 ,   0.4645  ,  0.5735  ,  0.5547,    0.6699  ,  0.3677 ,   0.6026 ,   0.5552  ,  0.5169];
from_scratch = [0.2752  ,  0.7497   , 0.7970  ,  0.8092  ,  0.8505   , 0.7791   , 0.7872  ,  0.7800  ,  0.8028 ,   0.8146];
acc = [0.4067  ,  0.5298  ,  0.5095   , 0.5316  ,  0.4887    ,0.5722   , 0.5455 ,   0.6607 ,   0.4601   , 0.4648];


figure('Color', 'w', 'Position', [100, 100, 1100, 700]);

plot(data_portions*100, vanilla, '-o', 'DisplayName', 'Decoder trained on BrainGPT features', 'Color', colors(3, :)); hold on;
plot(data_portions*100, wavelet, '-o', 'DisplayName', 'Decoder trained on Wavelet features', 'Color', colors(1, :));
plot(data_portions*100, from_scratch, '-x', 'DisplayName', 'Decoder trained on DINO BrainGPT features', 'Color', colors(2, :));
plot(data_portions*100, acc, '-o', 'DisplayName', 'Decoder trained on Accelerometer BrainGPT features', 'Color', colors(2, :));
plot(data_portions*100, finetuned, '-s', 'DisplayName', 'Decoder trained on finetuned BrainGPT with DINO', 'Color', colors(2, :)); hold on;

xlabel('Percentage of the training dataset the models were trained on');
ylabel('F1 scores');
title('F1 scores trained decoders on different features with different amounts of training data');
legend;
ylim([0 1]);

% name_png = "COMPARISON.png";
% name_svg = "COMPARISON.svg";
% 
% directory = '/home/cyberspace007/mpicek/master_project/decoder_testing/figures/';
% 
% full_path_png = fullfile(directory, name_png);
% full_path_svg = fullfile(directory, name_svg);
% 
% exportgraphics(gcf, full_path_png, 'Resolution', 300); % 300 DPI resolution for high quality
% print(full_path_svg, '-dsvg', '-r300', '-opengl');

%%
% PLOTTING AVERAGE ACCURACIES ON THE DIFFERENT SIZES OF THE TRAINING
% DATASET
figure;
errorbar(data_portions*100, accuracies_multimodal, stds_multimodal, '-o', 'DisplayName', 'Accuracies Multimodal BrainGPT features'); hold on;
errorbar(data_portions*100, accuracies_vanilla, stds_vanilla, '-o', 'DisplayName', 'Accuracies Vanilla BrainGPT features'); hold on;
errorbar(data_portions*100, accuracies_wavelet, stds_wavelet, '-o', 'DisplayName', 'Accuracies Wavelet features'); hold on;


% Adding labels and title
xlabel('Percentage of the training dataset the models were trained on');
ylabel('Accuracies');
title('Accuracies of different models based on portions of training data');

% Adding a legend
legend;

% Display the plot
grid on;
hold off;

%%
% PLOTTING THE PROBABILITIES OF A MOVEMENT TYPE

figure('Position', [100, 100, 1600, 800]);
set(gcf, 'Renderer', 'painters');
n_states = size(selected_states,2);
%colors = [NRColors.CoolGray5; NRColors.DarkYellow; NRColors.ArcBlue; NRColors.DeepOrange; NRColors.Magenta; NRColors.ImpulseRed] ; % This generates n distinct colors
%colors = [NRColors.CoolGray5; NRColors.DeepRed; NRColors.DeepMagenta; NRColors.DeepViolet; NRColors.DeepBlue; NRColors.DeepCyan] ; % This generates n distinct colors
colors = [...
    0.812, 0.816, 0.816; ...
    0.298, 0.008, 0.071;...
    0.969, 0.725, 0.690;...
    0.980, 0.322, 0.357;...
    0.788, 0.090, 0.173;...
    0.537, 0.027, 0.118;...
    0.298, 0.008, 0.071];

ax1 = subplot(3, 1, 1);
%plot(pred_wavelet, 'LineWidth',1)
hold on;
for i = 1:n_states
    plot(sel_pred_wavelet(:, i), 'Color', colors(i, :), 'LineWidth',1);
end
for i = 1:n_states
    % Find indices where cue is active
    cueOnsets = find(aligned_sel_u_test_wavelet(:, i) == 1);
    for j = 1:length(cueOnsets)
        areaX = [cueOnsets(j) cueOnsets(j) cueOnsets(j)+1 cueOnsets(j)+1];
        areaY = [0 1 1 0]; % Assuming probabilities are between 0 and 1
        fill(areaX, areaY, colors(i, :), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end
xlabel('Time (ds)');
ylabel('Probability');
%title('Decoder based on the Wavelet Features')
legend('rest', 'shoulder abduction', 'elbow extension', 'pronation', 'hand open', 'hand close'); % Update accordingly
hold off;
title('Decoder based on the Wavelet Features')


ax2 = subplot(3, 1, 2);
%plot(pred_wavelet, 'LineWidth',1)
hold on;
for i = 1:n_states
    plot(sel_pred_vanilla(:, i), 'Color', colors(i, :), 'LineWidth',1);
end
for i = 1:n_states
    % Find indices where cue is active
    cueOnsets = find(aligned_sel_u_test_vanilla(:, i) == 1);
    for j = 1:length(cueOnsets)
        areaX = [cueOnsets(j) cueOnsets(j) cueOnsets(j)+1 cueOnsets(j)+1];
        areaY = [0 1 1 0]; % Assuming probabilities are between 0 and 1
        fill(areaX, areaY, colors(i, :), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end
xlabel('Time (ds)');
ylabel('Probability');
%legend('State 1', 'State 2', ..., 'State N'); % Update accordingly
hold off;
title('Decoder based on the BrainGPT Features')

ax3 = subplot(3, 1, 3);
%plot(pred_wavelet, 'LineWidth',1)
hold on;
for i = 1:n_states
    plot(sel_pred_multimodal(:, i), 'Color', colors(i, :), 'LineWidth',1);
end
for i = 1:n_states
    % Find indices where cue is active
    cueOnsets = find(aligned_sel_u_test_multimodal(:, i) == 1);
    for j = 1:length(cueOnsets)
        areaX = [cueOnsets(j) cueOnsets(j) cueOnsets(j)+1 cueOnsets(j)+1];
        areaY = [0 1 1 0]; % Assuming probabilities are between 0 and 1
        fill(areaX, areaY, colors(i, :), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end
xlabel('Time (ds)');
ylabel('Probability');
%legend('State 1', 'State 2', ..., 'State N'); % Update accordingly
hold off;
title('Decoder based on the Multimodal BrainGPT Features')

linkaxes([ax1 ax2 ax3], 'x')

% TO SAVE THE FIGURES, UNCOMMENT THE FOLLOWING CODE:
name_png = "PROBABILITIES_DINO_SSL_full_retrain_29_100_percent_data.png";
name_svg = "PROBABILITIES_DINO_SSL_full_retrain_29_100_percent_data.svg";

directory = '/home/cyberspace007/mpicek/master_project/decoder_testing/figures/';

% Create the full path by concatenating the directory and the filename
full_path_png = fullfile(directory, name_png);
full_path_svg = fullfile(directory, name_svg);

exportgraphics(gcf, full_path_png, 'Resolution', 300); % 300 DPI resolution for high quality
print(full_path_svg, '-dsvg', '-r300', '-opengl');

%%
% PLOTTING THE CONFUSION MATRICES

index = length(data_portions);
% index = 1;

% calculate confusion matrix
%thresh_pred_wavelet = (sel_pred_wavelet==max(sel_pred_wavelet, [], 2));
%acc_wavelet=thresh_pred_wavelet'*aligned_sel_u_test_wavelet;
acc_wavelet = squeeze(all_confusion_matrices_wavelet(index, :, :));
acc_wavelet=100*acc_wavelet./sum(aligned_sel_u_test_wavelet);

%thresh_pred_multimodal = (sel_pred_multimodal==max(sel_pred_multimodal, [], 2));
%acc_multimodal=thresh_pred_multimodal'*aligned_sel_u_test_multimodal;
acc_multimodal = squeeze(all_confusion_matrices_multimodal(index, :, :));
acc_multimodal=100*acc_multimodal./sum(aligned_sel_u_test_multimodal);

%thresh_pred_vanilla = (sel_pred_vanilla==max(sel_pred_vanilla, [], 2));
%acc_vanilla=thresh_pred_vanilla'*aligned_sel_u_test_vanilla;
acc_vanilla = squeeze(all_confusion_matrices_vanilla(index, :, :));
acc_vanilla=100*acc_vanilla./sum(aligned_sel_u_test_vanilla);


figure('Position', [100, 100, 1400, 500]);
ax1 = subplot(1,3,1);
imagesc(acc_wavelet);
colormap(nr_red_white_map); % Choose a colormap as per your preference
colorbar;
clim([0 100]);
ylabel('Predicted Labels');
xlabel('True Labels');

% move the title a tiny bit up
titleHandle = title('Wavelet Features');
set(titleHandle, 'Units', 'normalized');
titlePosition = get(titleHandle, 'Position');
titlePosition(1) = titlePosition(1) + 0.05; 
titlePosition(2) = titlePosition(2) + 0.05; 
set(titleHandle, 'Position', titlePosition);

for i = 1:n_states
    for j = 1:n_states
        accuracyStr = sprintf('%.1f%', acc_wavelet(i,j));
        text(j, i, accuracyStr, 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
    end
end
axis square;
set(gca, 'XTick', 1:n_states, 'YTick', 1:n_states);

ax2 = subplot(1,3,2);
imagesc(acc_vanilla);
colormap(nr_red_white_map); % Choose a colormap as per your preference
colorbar;
clim([0 100]);
ylabel('Predicted Labels');
xlabel('True Labels');

% move the title a tiny bit up
titleHandle = title('BrainGPT Features');
set(titleHandle, 'Units', 'normalized');
titlePosition = get(titleHandle, 'Position');
titlePosition(1) = titlePosition(1) + 0.05; 
titlePosition(2) = titlePosition(2) + 0.05; 
set(titleHandle, 'Position', titlePosition);

for i = 1:n_states
    for j = 1:n_states
        accuracyStr = sprintf('%.1f%', acc_vanilla(i,j));
        text(j, i, accuracyStr, 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
    end
end
axis square;
set(gca, 'XTick', 1:n_states, 'YTick', 1:n_states);

ax3 = subplot(1,3,3);
imagesc(acc_multimodal);
colormap(nr_red_white_map); % Choose a colormap as per your preference
colorbar;
clim([0 100]);
ylabel('Predicted Labels');
xlabel('True Labels');

% move the title a tiny bit up
titleHandle = title('Multimodal BrainGPT Features');
set(titleHandle, 'Units', 'normalized');
titlePosition = get(titleHandle, 'Position');
titlePosition(1) = titlePosition(1) + 0.05; 
titlePosition(2) = titlePosition(2) + 0.05; 
set(titleHandle, 'Position', titlePosition);

for i = 1:n_states
    for j = 1:n_states
        accuracyStr = sprintf('%.1f%', acc_multimodal(i,j));
        text(j, i, accuracyStr, 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
    end
end
axis square;
set(gca, 'XTick', 1:n_states, 'YTick', 1:n_states);

% % TO SAVE THE FIGURES, UNCOMMENT THE FOLLOWING CODE:
% name_png = "CONF_MATRIX_DINO_SSL_full_retrain_29_30_percent_data.png";
% name_svg = "CONF_MATRIX_DINO_SSL_full_retrain_29_30_percent_data.svg";
% 
% directory = '/home/cyberspace007/mpicek/master_project/decoder_testing/figures/';
% 
% full_path_png = fullfile(directory, name_png);
% full_path_svg = fullfile(directory, name_svg);
% 
% exportgraphics(gcf, full_path_png, 'Resolution', 300); % 300 DPI resolution for high quality
% print(full_path_svg, '-dsvg', '-r300', '-opengl');


%%
% PLOTTING THE T-SNE PLOTS
% I SAVE THE PLOT MANUALLY (I SAVE IT TO .FIG IN THE FIGURE VIEWER AND THEN
% LATER IN MATLAB COMMAND WINDOW I SAVE IT TO .PNG AND .SVG)

tsne_wavelet = tsne(sel_x(aligned_sel_u_test_wavelet(:,1) ~= 1, :));
tsne_multimodal = tsne(sel_xLatent_multimodal(aligned_sel_u_test_multimodal(:,1) ~= 1, :));
tsne_vanilla = tsne(sel_xLatent_vanilla(aligned_sel_u_test_vanilla(:,1) ~= 1, :));

% colors = [NRColors.CoolGray5; NRColors.FadeOrange; NRColors.ArcBlue; NRColors.DarkRed; NRColors.Magenta; NRColors.ImpulseRed] ; % This generates n distinct colors
colors_cmap = ["#cfd0d0"; "#b1b2b3"; "#f7b9b0"; "#FA525B"; "#C9172C"; "#89071E"; "#4C0212"];
colors = [...
    0.812, 0.816, 0.816; ...
    0.298, 0.008, 0.071;...
    0.969, 0.725, 0.690;...
    0.980, 0.322, 0.357;...
    0.788, 0.090, 0.173;...
    0.537, 0.027, 0.118;...
    0.298, 0.008, 0.071];
point_size = 8;

% plot(probaPredicted')

figure('Color', 'W');
ax1 = subplot(1,3,1);
hold on;
h = gobjects(n_states-1, 1); % Preallocate graphics object array for legend
for i = 2:n_states
    % Extract data points corresponding to each label
    label_data = tsne_wavelet(aligned_sel_u_test_wavelet(aligned_sel_u_test_wavelet(:,1) ~= 1,i) == 1, :);
    
    % Scatter plot for each label
    h(i-1) = scatter(label_data(:,1), label_data(:,2), point_size, "MarkerEdgeColor", colors(i, :), ...
    "MarkerFaceColor", colors(i, :));
end
hold off;

title({'t-SNE of the Wavelet', 'space'}, 'Interpreter', 'tex');
xlabel('t-SNE 1');
ylabel('t-SNE 2');
axis square;
set(gca, 'XTickLabel', [], 'YTickLabel', [], 'XColor', 'none', 'YColor', 'none'); % Hide axis numbers

ax2 = subplot(1, 3,2);
hold on;
for i = 2:n_states
    % Extract data points corresponding to each label
    label_data = tsne_vanilla(aligned_sel_u_test_vanilla(aligned_sel_u_test_vanilla(:,1) ~= 1,i) == 1, :);
    
    % Scatter plot for each label
    scatter(label_data(:,1), label_data(:,2), point_size, "MarkerEdgeColor", colors(i, :), ...
    "MarkerFaceColor", colors(i, :));
end
hold off;
title({'t-SNE of the', 'BrainGPT space'}, 'Interpreter', 'tex');
xlabel('t-SNE 1');
ylabel('t-SNE 2');
axis square;
set(gca, 'XTickLabel', [], 'YTickLabel', [], 'XColor', 'none', 'YColor', 'none'); % Hide axis numbers

ax3 = subplot(1,3,3);
hold on;
for i = 2:n_states
    % Extract data points corresponding to each label
    label_data = tsne_multimodal(aligned_sel_u_test_multimodal(aligned_sel_u_test_multimodal(:,1) ~= 1,i) == 1, :);
    
    % Scatter plot for each label
    scatter(label_data(:,1), label_data(:,2), point_size, "MarkerEdgeColor", colors(i, :), ...
    "MarkerFaceColor", colors(i, :));
end
hold off;
title({'t-SNE of the', 'Multimodal BrainGPT space'}, 'Interpreter', 'tex');
xlabel('t-SNE 1');
ylabel('t-SNE 2');
axis square;
set(gca, 'XTickLabel', [], 'YTickLabel', [], 'XColor', 'none', 'YColor', 'none'); % Hide axis numbers

% Create a single legend
lgd = legend(h, {'shoulder abduction', 'elbow extension', 'pronation', 'hand open', 'hand close'}, ...
    'Location', 'eastoutside', 'Orientation', 'vertical');
set(lgd, 'Box', 'off');